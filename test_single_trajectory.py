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

    # --- DEBUGGING SCRIPT (Please remove or comment out these lines for full runs) ---
#    if global_traj_idx == 0:
#        print(f"\n--- [New2 Code] DEBUGGING TRAJECTORY {global_traj_idx} ---")
#        print("Initial State Vector:", initial_trajectory_data)
#        
#        derivs = qsys.derivs(0, initial_trajectory_data)
#        print("Derivatives at First Step:", derivs)
#        
#        print("--- Test complete. Aborting. ---\n")
#        sys.stdout.flush()
#        MPI.COMM_WORLD.Abort(0)

    return TrajectoryData(
        raw_diabatic_pops_vs_time=np.array([]),
        raw_adiabatic_pops_vs_time=np.array([]),
        E_total_vs_time=np.array([]),
        is_bad_trajectory=True,
        original_trajectory_index=global_traj_idx
    )
