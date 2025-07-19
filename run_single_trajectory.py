import numpy as np
from scipy.integrate import solve_ivp
from initialize_trajectory import initialize_traj
from derivs_dia import rho_elec, derivs_meyer_miller, derivs_nuclear
from nucl_dia_H_and_dH import SpinBoson
from transformations import (nq_from_xp, unflatten, flatten,
                             unflatten_solution_array_all_times)
from analysis import TrajectoryData
from window_options import get_triangular_population
from mpi4py import MPI
import sys # Import sys to use flush

class SystemForSolver:
    def __init__(self, params):
        self.F = int(params.get("F", 2))
        self.n_modes = int(params.get("n_modes", 100))
        self.m = params["built_bath_params"]["bath_mass"]
        if params.get("window_model", "").lower() == "triangular":
            self.L = 1.0 / 3.0
        else:
            self.L = params.get("L", 0.366)
        self.sb_model = SpinBoson(params, params["built_bath_params"])
        self._U_at_previous_step = np.eye(self.F)

    def derivs(self, t, trajectory):
        trajectory_x, trajectory_p, trajectory_R, trajectory_P = \
            unflatten(trajectory, self.F, self.n_modes)
        H_diab = self.sb_model.H(trajectory_R)
        rho_matrix = rho_elec(trajectory_x, trajectory_p, self.L, self.F)
        # Removed the unnecessary call to get_adiabatic_properties for diabatic dynamics
        # _, U_current = self.sb_model.get_adiabatic_properties(trajectory_R, self._U_at_previous_step)
        # self._U_at_previous_step = U_current
        dx_dt_elec, dp_dt_elec = derivs_meyer_miller(
             H_diab, trajectory_x, trajectory_p, self.F)
        dR_dt_nucl, dP_dt_nucl = derivs_nuclear(
            self.sb_model, rho_matrix,
            trajectory_R, trajectory_P, self.m, self.n_modes)
        return flatten(dx_dt_elec, dp_dt_elec, dR_dt_nucl, dP_dt_nucl)

# Accept rng object as a parameter
def run_single_traj(params, global_traj_idx, rng):
    qsys = SystemForSolver(params)
    
    # Removed re-seeding for each trajectory; now uses the rank-specific rng
    # rng = np.random.default_rng(global_traj_idx)
    initial_trajectory_data = initialize_traj(params, rng)

    #--- DEBUGGING SCRIPT (Please remove or comment out these lines for full runs) ---
    if global_traj_idx == 0:
         print(f"\n--- [New2 Code] DEBUGGING TRAJECTORY {global_traj_idx} ---")
         print("Initial State Vector:", initial_trajectory_data)
        
         derivs = qsys.derivs(0, initial_trajectory_data)
         print("Derivatives at First Step:", derivs)
        
         print("--- Test complete. Aborting. ---\n")
         sys.stdout.flush()
#         MPI.COMM_WORLD.Abort(0)
    # --- END DEBUGGING SCRIPT ---

    # --- RESTORED: Actual simulation integration and processing logic ---
    end_t = float(params.get("end_time", 1.0))
    n_t_out = int(params.get("n_times", 100))
    t_eval_points = np.linspace(0, end_t, n_t_out)

    traj_data_obj = TrajectoryData(np.array([]), np.array([]), np.array([]), is_bad_trajectory=True, original_trajectory_index=global_traj_idx)

    try:
        y0_f = initial_trajectory_data # Initial state is already flattened and ordered correctly

        sol = solve_ivp(
            qsys.derivs, (0, end_t), y0_f,
            method='DOP853', t_eval=t_eval_points,
            #args=(qsys,),
            atol=params.get("ode_atol", 1e-8), rtol=params.get("ode_rtol", 1e-8)
        )
        
        # --- ADDED: Debugging print statement ---
        print(f"Rank {MPI.COMM_WORLD.Get_rank()}: Traj {global_traj_idx} (Simplified Code) finished with {sol.nfev} function evaluations.")

        if sol.success:
            # Process results: calculate populations and energies
            N_nucl, F_elec = qsys.sb_model.bath_N, qsys.F
            # unflatten_solution_array_all_times needs to be adjusted to return order [x,p,R,P] or handle the [R,P,x,p] order
            # The current transformation.py's unflatten_solution_array_all_times already returns (x,p,R,P)
            # and expects (R,P,x,p) as input, so this should be compatible.
            path_x, path_p, path_R, path_P = unflatten_solution_array_all_times(sol.y.T, F_elec, N_nucl) 
            
            # Calculate actions for population analysis
            actions_dia = 0.5 * (path_x**2 + path_p**2) - qsys.L
            raw_diab_pops_vs_time = np.zeros((n_t_out, F_elec))
            
            # Use get_triangular_population from window_options.py
            for k_t in range(n_t_out):
                raw_diab_pops_vs_time[k_t, :] = get_triangular_population(actions_dia[k_t, :], F_elec)

            # E_total_vs_time needs to be calculated if desired for full output
            # For simplicity, we can initialize to zeros if not directly calculating total energy here for each time step in this simplified version.
            # If a full calculation is needed, qsys.H_total_system needs to be called per time step
            # For now, initializing adiabatic populations and total energy as zeros for this simplified diabatic run.
            raw_adia_pops_vs_time = np.zeros_like(raw_diab_pops_vs_time)
            E_total_vs_time = np.zeros(n_t_out) # Placeholder for now

            traj_data_obj = TrajectoryData(
                raw_diabatic_pops_vs_time=raw_diab_pops_vs_time,
                raw_adiabatic_pops_vs_time=raw_adia_pops_vs_time, # Typo in original code's variable name if copied, fixed.
                E_total_vs_time=E_total_vs_time,
                is_bad_trajectory=False,
                original_trajectory_index=global_traj_idx
            )
        else:
            print(f"Warning: Trajectory {global_traj_idx} failed to integrate. Message: {sol.message}")
            traj_data_obj.is_bad_trajectory = True
    except Exception as e:
        print(f"Warning: Trajectory {global_traj_idx} failed with error: {e}")
        traj_data_obj.is_bad_trajectory = True
        
    return traj_data_obj
