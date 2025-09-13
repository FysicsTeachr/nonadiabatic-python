# simulation/run_single_trajectory_qm.py
import numpy as np
import sys
from pathlib import Path

from models.qm import QM
from simulation.initialize_trajectory import initialize_traj
from utils.transformations import unflatten, flatten
from analysis.data_types import TrajectoryData
from models.window_options import get_histogram_population
from utils.transformations import unflatten_solution_array_all_times, nq_from_xp
from .derivs import derivs_meyer_miller_adiabatic, get_nuclear_derivatives

class SystemForSolverQM:
    """
    VV with RK4 substeps.
    """
    def __init__(self, params, qm_model):
        self.params = params
        self.F = int(params["F"])
        self.nuclear_model = qm_model
        self.L = float(self.params.get("L", 0.5))
        
        self.current_adiab_E = np.zeros(self.F)
        n_nucl_coords = self.nuclear_model.n_atoms * 3
        self.current_nac_tensor = np.zeros((self.F, self.F, n_nucl_coords))
        self.current_Hel_dR_adia_list = []
        self.E_pot_initial = 0.0 # Store initial potential energy

    def get_electronic_derivs(self, x, p, R_dot_flat):
        return derivs_meyer_miller_adiabatic(
            self.current_adiab_E, self.current_nac_tensor, x, p, R_dot_flat
        )

    def update_qm_and_get_nuclear_derivs(self, x, p, R, P, step_idx):
        raw_adiab_E, Hel_dR, nac_tensor = self.nuclear_model.get_qm_properties(R, step_idx)
        
        # --- FIX: Set initial energy only on the first step ---
        if step_idx == 0:
            self.E_pot_initial = raw_adiab_E[0]

        self.current_adiab_E = raw_adiab_E - self.E_pot_initial
        # --- END FIX ---
        
        self.current_nac_tensor = nac_tensor
        self.current_Hel_dR_adia_list = Hel_dR
        
        dR_dt, dP_dt = get_nuclear_derivatives(
            x, p, P, self.current_Hel_dR_adia_list, self.nuclear_model, 
            self.L, self.current_adiab_E, self.current_nac_tensor
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

def calculate_energy_components(y_vec, adiab_E, masses, F, L, n_atoms):
    x, p, R, P = unflatten(y_vec, F, n_atoms)
    E_kin_nuc = np.sum(0.5 * P**2 / masses[:, np.newaxis])
    
    n = 0.5 * (x**2 + p**2) - L
    mod = (1.0 - np.sum(n)) / F if F > 0 else 0.0
    C_diag = n + mod
    V_elec_exp = np.sum(C_diag * adiab_E)
    
    E_total = E_kin_nuc + V_elec_exp
    return E_total, E_kin_nuc, V_elec_exp

def run_single_traj_qm(params, global_traj_idx, rng):
    qm_model = None
    try:
        qm_model = QM(params)
        qsys = SystemForSolverQM(params, qm_model)

        y = initialize_traj(params, qsys.nuclear_model, rng)
        dt = params["dt_au"]
        n_steps = int(params["end_time_au"] / dt)
        n_t_out = n_steps + 1
        
        y_all_steps = np.zeros((n_t_out, len(y)))
        y_all_steps[0, :] = y
       
        adiabatic_pops = np.zeros((n_t_out, qsys.F))
        E_total_vs_time = np.zeros(n_t_out)
        E_kin_vs_time = np.zeros(n_t_out)
        E_pot_vs_time = np.zeros(n_t_out)

        x, p, R, P = unflatten(y, qsys.F, qsys.nuclear_model.n_atoms)
        x = np.atleast_1d(x)
        p = np.atleast_1d(p)

        actions_init, _ = nq_from_xp(x, p, qsys.L)
        adiabatic_pops[0, :] = get_histogram_population(actions_init, qsys.L, qsys.F)
        
        print("\n" + "="*50)
        print(f"STARTING TRAJECTORY {global_traj_idx}")
        print("="*50)

        _, P_dot = qsys.update_qm_and_get_nuclear_derivs(x, p, R, P, 0)
        
        E_total, E_kin, E_pot = calculate_energy_components(y, qsys.current_adiab_E, qsys.nuclear_model.masses, qsys.F, qsys.L, qsys.nuclear_model.n_atoms)
        E_total_vs_time[0] = E_total
        E_kin_vs_time[0] = E_kin
        E_pot_vs_time[0] = E_pot
        
        for k_t in range(1, n_t_out):
            print(f"\n{'='*20} PROPAGATION STEP {k_t} {'='*20}")
            
            P_half = P + 0.5 * P_dot * dt
            masses_reshaped = qsys.nuclear_model.masses[:, np.newaxis]
            R_new = R + (P_half / masses_reshaped) * dt
            
            R_dot_flat = (P_half / masses_reshaped).flatten()
            x_new, p_new = rk4_step_electronic(x, p, R_dot_flat, qsys, dt)
            
            _, P_dot_new = qsys.update_qm_and_get_nuclear_derivs(x_new, p_new, R_new, P_half, k_t)
            
            P_new = P_half + 0.5 * P_dot_new * dt
            x, p, R, P, P_dot = x_new, p_new, R_new, P_new, P_dot_new
            y = flatten(x, p, R, P)
            y_all_steps[k_t, :] = y
            
            actions, _ = nq_from_xp(x, p, qsys.L)
            adiabatic_pops[k_t, :] = get_histogram_population(actions, qsys.L, qsys.F)
            
            E_total, E_kin, E_pot = calculate_energy_components(y, qsys.current_adiab_E, qsys.nuclear_model.masses, qsys.F, qsys.L, qsys.nuclear_model.n_atoms)
            E_total_vs_time[k_t] = E_total
            E_kin_vs_time[k_t] = E_kin
            E_pot_vs_time[k_t] = E_pot

        return TrajectoryData(
            raw_diabatic_pops_vs_time=np.array([]),
            raw_adiabatic_pops_vs_time=adiabatic_pops,
            E_total_vs_time=E_total_vs_time,
            E_kin_vs_time=E_kin_vs_time,
            E_pot_vs_time=E_pot_vs_time,
            is_bad_trajectory=False,
            original_trajectory_index=global_traj_idx
        )

    except Exception as e:
        print(f"FATAL ERROR: Trajectory {global_traj_idx} crashed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return TrajectoryData(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), True, global_traj_idx)
    finally:
        if qm_model:
            qm_model.close()
