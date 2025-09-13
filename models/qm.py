# models/qm.py
import numpy as np
import time
import sys
import parmed as pmd
import pytcpb as tc
#import os
#print(f"--- Using pytcpb from: {os.path.dirname(tc.__file__)}")
class QM:
    """
    An interface to TCPB via the pytcpb wrapper to perform direct dynamics 
    QM/MM simulations.
    """
    def __init__(self, params):
        self.params = params
        self.F = int(params["F"])
        self.n_atoms = int(params["n_atoms"])
        self.n_qm_atoms = int(params["n_qm_atoms"])

        hostname = params["tcpb_hostname"]
        port = int(params["tcpb_port"])

        # Connect using the pytcpb wrapper
        status = tc.connect(host=hostname, port=port)
        if status != 0:
            raise ConnectionError(f"Failed to connect to TeraChem server at {hostname}:{port}")

        print(f"QM Model Initialized: Connected to TCPB server at {hostname}:{port}")

        self.masses = None
        self.atom_symbols = None
        self.qmattypes = None
        self.qm_indices = None
        self.mm_indices = None
        self.mm_charges = None
        
        self._setup_system_from_amber_files()

    def close(self):
        """Finalizes the TeraChem server connection."""
        tc.finalize()
        print("TeraChem connection finalized.")

    def _setup_system_from_amber_files(self):
        """
        Loads system topology and sets up the TeraChem server via pytcpb.
        """
        prmtop_file = self.params["prmtop_file"]
        rst7_file = self.params["rst7_file"]
        qmregion_file = self.params["qmregion_file"]

        with open(qmregion_file, 'r') as f:
            self.qm_indices = {int(line.strip()) for line in f if line.strip()}

        structure = pmd.load_file(prmtop_file, rst7_file)
        if self.n_atoms != len(structure.atoms):
            print(f"Warning: n_atoms in argm file ({self.n_atoms}) does not match prmtop ({len(structure.atoms)}). Using prmtop value.")
            self.n_atoms = len(structure.atoms)
            self.params['n_atoms'] = self.n_atoms
        
        self.masses = np.array([atom.mass for atom in structure.atoms])

        qmattypes = []
        mmcharges = []
        mm_indices = []

        for i, atom in enumerate(structure.atoms):
            if i in self.qm_indices:
                qmattypes.append(atom.element_name)
            else:
                mmcharges.append(atom.charge)
                mm_indices.append(i)

        self.qmattypes = qmattypes
        self.mm_charges = np.array(mmcharges)
        self.mm_indices = np.array(mm_indices, dtype=int)
        self.qm_indices = np.array(list(self.qm_indices), dtype=int)
        
        tcfile = self.params.get("terachem_input_file", "terachem.inp")
        status = tc.setup(tcfile, self.qmattypes)
        if status != 0:
            raise RuntimeError(f"TeraChem setup failed with status {status}")
        print("TeraChem calculation successfully set up via pytcpb.")


    def get_qm_properties(self, R_coords_bohr, step_idx):
        """
        Computes QM/MM properties using the pytcpb wrapper.
        """
        qm_coords = R_coords_bohr[self.qm_indices].flatten().tolist()
        mm_coords = R_coords_bohr[self.mm_indices].flatten().tolist()
        mm_charges_list = self.mm_charges.tolist()
        
        energy, qm_grad_au, mm_grad_au, status = tc.compute_energy_gradient(
            self.qmattypes, qm_coords, mm_coords, mm_charges_list, 0
        )

        if status != 0:
            raise RuntimeError(f"TeraChem compute_energy_gradient failed at step {step_idx} with status {status}")

        gradients = np.zeros((self.n_atoms, 3))
        gradients[self.qm_indices] = np.array(qm_grad_au).reshape(self.n_qm_atoms, 3)
        gradients[self.mm_indices] = np.array(mm_grad_au).reshape(len(self.mm_indices), 3)

        # --- Step 2: Compute Non-Adiabatic Coupling ---
        # This new function call computes the coupling using the same geometry
#        coupling, status = tc.compute_coupling(
#            self.qmattypes, qm_coords, mm_coords, mm_charges_list, 0
#        )


        adiab_E = np.array([energy])
        nac_tensor = np.zeros((1, 1, self.n_atoms * 3))
        gradients_flat = gradients.flatten()
        Hel_dR_adia_list = []
        for k in range(self.n_atoms * 3):
            mat = np.array([[-gradients_flat[k]]])
            Hel_dR_adia_list.append(mat)

        return adiab_E, Hel_dR_adia_list, nac_tensor

    def initialize_nuclear_coordinates(self, rng):
        prmtop_file = self.params["prmtop_file"]
        rst7_file = self.params["rst7_file"]
        structure = pmd.load_file(prmtop_file, rst7_file)
        
        R_angstrom = np.array(structure.coordinates)
        P = np.zeros_like(R_angstrom)
        
        AMU_TO_AU = 1822.888486
        ANGSTROM_TO_BOHR = 1.889726
        
        masses_au = self.masses * AMU_TO_AU
        R_bohr = R_angstrom * ANGSTROM_TO_BOHR
        
        self.masses = masses_au
        return R_bohr, P
