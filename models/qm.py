# models/qm.py
import numpy as np
import time
import sys
import parmed as pmd
import pytcpb as tc
import copy
import ctypes
from pathlib import Path

try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
except ImportError:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *

# CTYPES SETUP for Custom Coupling Function
tc.libtcpb.coupling_using_previous_mm.argtypes = (
    ctypes.c_char_p,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int)
)
tc.libtcpb.coupling_using_previous_mm.restype = None

class QM:
    def __init__(self, params):
        self.params = params
        self.F = int(params["F"])
        self.n_atoms = int(params["n_atoms"])
        self.n_qm_atoms = int(params["n_qm_atoms"])
        
        # --- FIX: Define rst path and ensure it exists ---
        self.rst_path = Path("rst")
        self.rst_path.mkdir(exist_ok=True)
        # --- END FIX ---
        
        hostname, port = params["tcpb_hostname"], int(params["tcpb_port"])
        if tc.connect(host=hostname, port=port) != 0:
            raise ConnectionError(f"Failed to connect to TeraChem server at {hostname}:{port}")
        print(f"QM Model Initialized: Connected to TCPB server at {hostname}:{port}")
        self.masses, self.qmattypes, self.qm_indices, self.mm_indices, self.mm_charges, self.mm_context, self.base_terachem_inp = [None] * 7
        self._setup_system_from_amber_files()
        self._generate_terachem_input_files()

    def close(self):
        tc.finalize()
        print("TeraChem connection finalized.")

    def _compute_custom_vector(self, runtype, qm_coords_bohr):
        result_out_type = ctypes.c_double * (3 * self.n_qm_atoms)
        result_out = result_out_type()
        status = ctypes.c_int()
        tc.libtcpb.coupling_using_previous_mm(runtype.encode('utf-8'), qm_coords_bohr.flatten(), ctypes.c_int(self.n_qm_atoms), result_out, ctypes.byref(status))
        return np.ctypeslib.as_array(result_out).reshape(self.n_qm_atoms, 3), status.value

    def _setup_system_from_amber_files(self):
        prmtop_file, rst7_file, qmregion_file = self.params["prmtop_file"], self.params["rst7_file"], self.params["qmregion_file"]
        with open(qmregion_file, 'r') as f:
            qm_indices_set = {int(line.strip()) for line in f if line.strip()}
        
        full_structure = pmd.load_file(prmtop_file, rst7_file)
        self.n_atoms = len(full_structure.atoms)
        self.params['n_atoms'] = self.n_atoms
        self.masses = np.array([atom.mass for atom in full_structure.atoms])
        
        qmattypes, mmcharges, mm_indices_list = [], [], []
        for i, atom in enumerate(full_structure.atoms):
            if i in qm_indices_set:
                qmattypes.append(atom.element_name)
            else:
                mmcharges.append(atom.charge)
                mm_indices_list.append(i)

        self.qmattypes, self.mm_charges = qmattypes, np.array(mmcharges)
        self.mm_indices, self.qm_indices = np.array(mm_indices_list, dtype=int), np.array(list(qm_indices_set), dtype=int)
        
        with open(self.params.get("terachem_input_file", "terachem.inp"), 'r') as f:
            self.base_terachem_inp = f.read()

        print("Creating a dedicated MM-only system for OpenMM...")
        mm_structure = copy.deepcopy(full_structure)
        qm_mask = f"@{','.join(map(str, [i+1 for i in self.qm_indices]))}"
        mm_structure.strip(qm_mask)
        system = mm_structure.createSystem(nonbondedMethod=NoCutoff)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.001*picoseconds) ##VerletIntegrator(0.001*picoseconds)
        self.mm_context = Context(system, integrator)
        print("OpenMM context for pure MM forces created successfully.")

    def _generate_terachem_input_files(self):
        print("Generating TeraChem input files in rst/ directory...")
        # --- FIX: Use the rst_path to create files in the correct directory ---
        with open(self.rst_path / "gs.inp", 'w') as f: f.write(self.base_terachem_inp)
        if self.F > 1:
            for i in range(1, self.F):
                with open(self.rst_path / f"es_{i}.inp", 'w') as f: f.write(f"{self.base_terachem_inp}\ncis yes\ncisnumstates {self.F}\ncistarget {i}\n")
            for i in range(self.F):
                for j in range(i + 1, self.F):
                    with open(self.rst_path / f"nac_{i}_{j}.inp", 'w') as f: f.write(f"{self.base_terachem_inp}\ncis yes\ncisnumstates {self.F}\nnacstate1 {i}\nnacstate2 {j}\n")
        # --- END FIX ---
        print("Input files generated.")

    def get_qm_properties(self, R_coords_bohr, step_idx):
        BOHR_TO_ANGSTROM = 0.52917721092
        ANGSTROM_TO_NM = 0.1
        HARTREE_TO_KJ_PER_MOL = 2625.5
        BOHR_TO_NM = BOHR_TO_ANGSTROM * ANGSTROM_TO_NM
        HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM = (HARTREE_TO_KJ_PER_MOL * AVOGADRO_CONSTANT_NA.value_in_unit(mole**-1)) / (BOHR_TO_NM)
        
        qm_coords_bohr, mm_coords_bohr = R_coords_bohr[self.qm_indices], R_coords_bohr[self.mm_indices]
        adiab_E, Hel_dR_adia_list, nac_tensor = np.zeros(self.F), [np.zeros((self.F, self.F)) for _ in range(self.n_atoms * 3)], np.zeros((self.F, self.F, self.n_atoms * 3))

        mm_coords_nm = mm_coords_bohr * BOHR_TO_ANGSTROM * ANGSTROM_TO_NM
        self.mm_context.setPositions(mm_coords_nm)
        state = self.mm_context.getState(getForces=True)
        mm_forces = state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer)
        mm_grads = -mm_forces / HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM

        if step_idx < 2:
            print(f"--- DEBUG (models/qm.py): Data sent to TeraChem at step {step_idx} ---")
            print(f"QM Coords (first 2 atoms, Bohr):\n{qm_coords_bohr[:2]}")
            print(f"MM Coords (first 2 atoms, Bohr):\n{mm_coords_bohr[:2]}")
            print("----------------------------------------------------------")

        # --- FIX: Use the rst_path when setting up TeraChem jobs ---
        tc.setup(str(self.rst_path / "gs.inp"), self.qmattypes)
        # --- END FIX ---
        energy, qm_grad, mm_grad_qm, status = tc.compute_energy_gradient(
            self.qmattypes, qm_coords_bohr.flatten(), mm_coords_bohr.flatten(), self.mm_charges.tolist(), 0
        )
        if status != 0: raise RuntimeError(f"GS calculation failed with status {status}")
        
        adiab_E[0] = energy
        total_gradients = np.zeros((self.n_atoms, 3))
        total_gradients[self.qm_indices] = np.array(qm_grad).reshape(self.n_qm_atoms, 3)
        total_gradients[self.mm_indices] = np.array(mm_grad_qm).reshape(len(self.mm_indices), 3) + mm_grads.reshape(len(self.mm_indices), 3)
        
        for k, grad_val in enumerate(total_gradients.flatten()): Hel_dR_adia_list[k][0, 0] = -grad_val

        if self.F > 1:
            for i_es in range(1, self.F):
                # --- FIX: Use the rst_path when setting up TeraChem jobs ---
                tc.setup(str(self.rst_path / f"es_{i_es}.inp"), self.qmattypes)
                # --- END FIX ---
                energy, qm_grad, mm_grad_qm, status = tc.compute_energy_gradient(
                    self.qmattypes, qm_coords_bohr.flatten(), mm_coords_bohr.flatten(), self.mm_charges.tolist(), 0
                )
                if status != 0: raise RuntimeError(f"ES {i_es} gradient failed with status {status}")
                adiab_E[i_es] = energy
                es_total_grads = np.zeros((self.n_atoms, 3))
                es_total_grads[self.qm_indices] = np.array(qm_grad).reshape(self.n_qm_atoms, 3)
                es_total_grads[self.mm_indices] = np.array(mm_grad_qm).reshape(len(self.mm_indices), 3) + mm_grads.reshape(len(self.mm_indices), 3)
                for k, grad_val in enumerate(es_total_grads.flatten()): Hel_dR_adia_list[k][i_es, i_es] = -grad_val
            
            for i in range(self.F):
                for j in range(i + 1, self.F):
                    # --- FIX: Use the rst_path when setting up TeraChem jobs ---
                    tc.setup(str(self.rst_path / f"nac_{i}_{j}.inp"), self.qmattypes)
                    # --- END FIX ---
                    nac_vec, status = self._compute_custom_vector("coupling", qm_coords_bohr)
                    if status != 0: raise RuntimeError(f"NAC coupling {i}-{j} failed with status {status}")
                    full_nac_vec = np.zeros((self.n_atoms, 3)); full_nac_vec[self.qm_indices] = nac_vec
                    nac_tensor[i, j, :] = nac_tensor[j, i, :] = full_nac_vec.flatten()
        
        # Part here is now handled in run_single_trajectory_qm.py, 
        # so we return the raw energies from this function.
        return adiab_E, Hel_dR_adia_list, nac_tensor

    def initialize_nuclear_coordinates(self, rng):
        prmtop_file, rst7_file = self.params["prmtop_file"], self.params["rst7_file"]
        structure = pmd.load_file(prmtop_file, rst7_file)
        
        R_angstrom = np.array(structure.coordinates)
        P = np.zeros_like(R_angstrom)

        AMU_TO_AU, ANGSTROM_TO_BOHR = 1822.888486, 1.889726
        self.masses *= AMU_TO_AU
        R_bohr = R_angstrom * ANGSTROM_TO_BOHR
        return R_bohr, P
