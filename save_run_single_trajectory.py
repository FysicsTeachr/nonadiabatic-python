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
        # The _U_at_previous_step logic is not needed for the diabatic dynamics loop.

    def derivs(self, t, trajectory):
        trajectory_x, trajectory_p, trajectory_R, trajectory_P = \
            unflatten(trajectory, self.F, self.n_modes)

        # Calculate H and rho once per step.
        H_diab = self.sb_model.H(trajectory_R)
        rho_matrix = rho_elec(trajectory_x, trajectory_p, self.L, self.F)

        # --- THE FIX ---
        # The unnecessary and expensive call to get_adiabatic_properties has been removed.
        # The logic is now identical to the original code's diabatic simulation.
        
        dx_dt_elec, dp_dt_elec = derivs_meyer_miller(
             H_diab, trajectory_x, trajectory_p, self.F)
        
        dR_dt_nucl, dP_dt_nucl = derivs_nuclear(
            self.sb_model, rho_matrix,
            trajectory_R, trajectory_P, self.m, self.n_modes)

        return flatten(dx_dt_elec, dp_dt_elec, dR_dt_nucl, dP_dt_nucl)

def run_single_traj(params, global_traj_idx):
    qsys = SystemForSolver(params)

    initial_trajectory_data = initialize_traj(
        params, np.random.default_rng(global_traj_idx))

    end_t = float(params.get("end_time", 3.0))
    n_times = int(params.get("n_times", 100))
    time_points_eval = np.linspace(0, end_t, n_times)

    sol = solve_ivp(
        qsys.derivs,
        (0, end_t),
        initial_trajectory_data,
        method='DOP853',
        t_eval=time_points_eval,
        atol=params.get("ode_atol", 1e-8),
        rtol=params.get("ode_rtol", 1e-8)
    )

    print(f"Rank {MPI.COMM_WORLD.Get_rank()}: Traj {global_traj_idx} finished with {sol.nfev} function evaluations.")

    if not sol.success:
        print(f"Warning: ODE integration failed for trajectory "
              f"{global_traj_idx}. Message: {sol.message}")
        return TrajectoryData(np.array([]), np.array([]), np.array([]), True, global_traj_idx)

    F, n_modes, L = qsys.F, qsys.n_modes, qsys.L
    path_x, path_p, path_R, path_P = \
        unflatten_solution_array_all_times(sol.y.T, F, n_modes)

    path_n, path_q = nq_from_xp(path_x, path_p, L)

    diabatic_pops_per_time = []
    for k_t in range(n_times):
        pop_indicators = get_triangular_population(path_n[k_t, :], F)
        diabatic_pops_per_time.append(pop_indicators)
    raw_diabatic_pops_vs_time_single_traj = np.array(diabatic_pops_per_time)

    E_total_vs_time_single_traj = np.zeros(n_times)
    raw_adiabatic_pops_vs_time_single_traj = \
        np.zeros_like(raw_diabatic_pops_vs_time_single_traj)

    return TrajectoryData(
        raw_diabatic_pops_vs_time=raw_diabatic_pops_vs_time_single_traj,
        raw_adiabatic_pops_vs_time=raw_adiabatic_pops_vs_time_single_traj,
        E_total_vs_time=E_total_vs_time_single_traj,
        is_bad_trajectory=False,
        original_trajectory_index=global_traj_idx
    )

