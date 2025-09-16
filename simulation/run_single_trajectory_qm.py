# simulation/run_single_trajectory_qm.py
import numpy as np
import sys
from pathlib import Path

from models.qm import QM
from simulation.initialize_trajectory import initialize_traj
from utils.transformations import unflatten, flatten, nq_from_xp
from analysis.data_types import TrajectoryData
from models.window_options import get_histogram_population
from .derivs import derivs_meyer_miller_adiabatic, get_nuclear_derivatives_meyer_miller

class SystemForSolverQM:
#Class to prepare for for using the ODE solver    
    def __init__(self, params, qm_model):
        self.params = params
        self.F = int(params["F"])
        self.nuclear_model = qm_model
        self.L = float(self.params.get("L", 0.366))
        
        self.current_adiab_E = np.zeros(self.F)
        n_nucl_coords = self.nuclear_model.n_atoms * 3
        self.current_nac_tensor = np.zeros((self.F, self.F, n_nucl_coords))
        self.current_Hel_dR_adia_list = []

    def get_electronic_derivs(self, x, p, R_dot_flat):
        return derivs_meyer_miller_adiabatic(
         self.current_adiab_E, self.current_nac_tensor, x, p, R_dot_flat
        )

    def update_qm_and_get_nuclear_derivs(self, x, p, R, P, step_idx):
        raw_adiab_E, Hel_dR, nac_tensor = self.nuclear_model.get_qm_properties(R, step_idx)
        
        self.current_adiab_E = raw_adiab_E - raw_adiab_E[0]
        self.current_nac_tensor = nac_tensor
        self.current_Hel_dR_adia_list = Hel_dR
        
        dR_dt, dP_dt = get_nuclear_derivatives_meyer_miller(
            x, p, P, self.current_Hel_dR_adia_list, self.nuclear_model, self.L
        )
        return dR_dt, dP_dt

def rk4_step_electronic(x, p, R_dot_flat, qsys, dt):
    k1_dx, k1_dp = qsys.get_electronic_derivs(x, p, R_dot_flat)
    x2, p2 = x + 0.5 * dt * k1_dx, p + 0.5 * dt * k1_dp
    k2_dx, k2_dp = qsys.get_electronic_derivs(x2, p2, R_dot_flat)
    x3, p3 = x + 0.5 * dt * k2_dx, p + 0.5 * dt * k2_dp
    k3_dx, k3_dp = qsys.get_electronic_derivs(x3, p3, R_dot_flat)
    x4, p4 = x + dt * k3_dx, p + dt * k3_dp
    k4_dx, k4_dp = qsys.get_electronic_derivs(x4, p4, R_dot_flat)
    x_new = x + (dt / 6.0) * (k1_dx + 2*k2_dx + 2*k3_dx + k4_dx)
    p_new = p + (dt / 6.0) * (k1_dp + 2*k2_dp + 2*k3_dp + k4_dp)
    return x_new, p_new

def run_single_traj_qm(params, global_traj_idx, rng):
    qm_model = None
    try:
        qm_model = QM(params)
        qsys = SystemForSolverQM(params, qm_model)

        y = initialize_traj(params, qsys.nuclear_model, rng)
        dt = params["dt_au"]
        n_steps = int(params["end_time_au"] / dt)
        n_t_out = n_steps + 1
        
        adiabatic_pops = np.zeros((n_t_out, qsys.F))
        E_total_vs_time = np.zeros(n_t_out)
        x, p, R, P = unflatten(y, qsys.F, qsys.nuclear_model.n_atoms)
        x, p = np.atleast_1d(x), np.atleast_1d(p)

        actions_init, _ = nq_from_xp(x, p, qsys.L)
        adiabatic_pops[0, :] = get_histogram_population(actions_init, qsys.L, qsys.F)
        
        print(f"\nSTARTING TRAJECTORY {global_traj_idx}")
        
        # Initial force calculation
        _, P_dot = qsys.update_qm_and_get_nuclear_derivs(x, p, R, P, 0)
        
        # Initial energy calculation
        KE_nucl = 0.5 * np.sum(P**2 / qsys.nuclear_model.masses[:, np.newaxis])
        E_elec = 0.5 * np.sum(p**2 + x**2) + np.dot(adiabatic_pops[0, :], qsys.current_adiab_E)
        E_total_vs_time[0] = KE_nucl + E_elec

        for k_t in range(1, n_t_out):
            # ======================================================================
            # --- DEBUG PRINTOUT at the start of each step ---
            # ======================================================================
            print("\n" + "="*20 + f" PROPAGATION STEP {k_t} " + "="*20)
            print(f"Time: {k_t * params['dt_au'] * 0.0241888:.4f} fs")
            print(f"Electronic x: {x}")
            print(f"Electronic p: {p}")
            print(f"Adiabatic Pops: {adiabatic_pops[k_t-1, :]}")
            print(f"QM Atom 0 Coords (Bohr): {R[qsys.nuclear_model.qm_indices[0]]}")
            print(f"MM Atom 0 Coords (Bohr): {R[qsys.nuclear_model.mm_indices[0]]}")
            # ======================================================================

            # --- Velocity Verlet Step ---
            P_half = P + 0.5 * P_dot * dt
            masses_reshaped = qsys.nuclear_model.masses[:, np.newaxis]
            R_new = R + (P_half / masses_reshaped) * dt
            
            # --- RK4 Step for Electronics ---
            R_dot_flat = (P_half / masses_reshaped).flatten()
            x_new, p_new = rk4_step_electronic(x, p, R_dot_flat, qsys, dt)
            
            # --- Get new forces and finish VV step ---
            _, P_dot_new = qsys.update_qm_and_get_nuclear_derivs(x_new, p_new, R_new, P_half, k_t)
            P_new = P_half + 0.5 * P_dot_new * dt
            
            # --- Update state for next iteration ---
            x, p, R, P, P_dot = x_new, p_new, R_new, P_new, P_dot_new
            
            # --- Adiabatic Population Binning ---
            actions, _ = nq_from_xp(x, p, qsys.L)
            adiabatic_pops[k_t, :] = get_histogram_population(actions, qsys.L, qsys.F)
            
            # --- Total Energy Calculation ---
            KE_nucl = 0.5 * np.sum(P**2 / qsys.nuclear_model.masses[:, np.newaxis])
            E_elec = 0.5 * np.sum(p**2 + x**2) + np.dot(adiabatic_pops[k_t, :], qsys.current_adiab_E)
            E_total_vs_time[k_t] = KE_nucl + E_elec


        return TrajectoryData(
            raw_adiabatic_pops_vs_time=adiabatic_pops,
            E_total_vs_time=E_total_vs_time,
            is_bad_trajectory=False,
            original_trajectory_index=global_traj_idx
        )

    except Exception as e:
        print(f"FATAL ERROR: Trajectory {global_traj_idx} crashed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Return a TrajectoryData object that indicates failure
        return TrajectoryData(np.array([]), np.array([]), is_bad_trajectory=True, original_trajectory_index=global_traj_idx)
    finally:
        if qm_model:
            qm_model.close()
