# simulation/run_single_trajectory.py
import numpy as np
from scipy.integrate import solve_ivp
import sys

# --- Imports from the modular structure ---
from simulation.initialize_trajectory import initialize_traj
from simulation.derivs import (rho_elec, derivs_meyer_miller_diabatic,
                     derivs_nuclear_adiabatic, derivs_mm_adia_general,
                     derivs_mm_adia_2_site)
from models.fmo import FMO
from models.ishizaki import Ishizaki
from models.window_options import get_histogram_population
from utils.transformations import unflatten, flatten, unflatten_solution_array_all_times, nq_from_xp
from analysis.data_types import TrajectoryData

# --- Model Factory ---
MODEL_DISPATCH = {
    "fmo": FMO,
    "ishizaki": Ishizaki,
}

class SystemForSolver:
    """
    A class that provides the derivative function for the ODE solver,
    encapsulating the entire physical system.
    """
    def __init__(self, params: dict):
        self.params = params
        self.is_adiabatic = params.get("Adiabatic", False)
        self.nuclear_model_name = params.get("nuclear_model", "").lower()
        self.F = int(params.get("F", 2))
        self.n_modes = int(params.get("n_modes", 100))
        self.L = float(params.get("L", 0.366))

        if self.nuclear_model_name not in MODEL_DISPATCH:
            raise NotImplementedError(f"Nuclear model '{self.nuclear_model_name}' not supported.")
        model_class = MODEL_DISPATCH[self.nuclear_model_name]
        self.nuclear_model = model_class(params, params["built_bath_params"])

        self.U_prev = np.identity(self.F)

    def derivs(self, t: float, trajectory: np.ndarray) -> np.ndarray:
        """Calculates the time derivative of the entire state vector."""
        x, p, R, P = unflatten(trajectory, self.F, self.n_modes)

        if self.is_adiabatic:
            adiab_E, U = self.nuclear_model.get_adiabatic_properties(R, self.U_prev)
            self.U_prev = U
            dH_dR_diab_list = [self.nuclear_model.dH_dRi_k(R, i, k) for i in range(self.F) for k in range(self.n_modes)]
            Hel_dR_adia_list = self.nuclear_model.get_Hel_dR_adia(U, dH_dR_diab_list)
            dR_dt = P / self.nuclear_model.m
            rho_adia = rho_elec(x, p, self.L, self.F)
            _, dP_dt = derivs_nuclear_adiabatic(self.nuclear_model, rho_adia, Hel_dR_adia_list)

            if self.nuclear_model_name == 'ishizaki':
                nac_vectors_flat = self.nuclear_model.get_NAC_vectors(adiab_E, Hel_dR_adia_list)
                dx_dt, dp_dt = derivs_mm_adia_2_site(adiab_E, nac_vectors_flat, x, p, dR_dt.flatten())
            else:
                nac_tensor = self.nuclear_model.get_NAC_vectors(adiab_E, Hel_dR_adia_list)
                dx_dt, dp_dt = derivs_mm_adia_general(adiab_E, nac_tensor, x, p, dR_dt.flatten())
        else:
            rho_matrix = rho_elec(x, p, self.L, self.F)
            H_diab = self.nuclear_model.H(R)
            dx_dt, dp_dt = derivs_meyer_miller_diabatic(H_diab, x, p, self.F)
            dR_dt, dP_dt = self.nuclear_model.get_nuclear_derivs(rho_matrix, R, P)

        return flatten(dx_dt, dp_dt, dR_dt, dP_dt)

def run_single_traj(params: dict, global_traj_idx: int, rng: np.random.Generator) -> TrajectoryData:
    """
    Initializes, runs, and analyzes a single non-adiabatic trajectory.
    """
    qsys = SystemForSolver(params)
    y0_diab_f = initialize_traj(params, qsys.nuclear_model, rng)

    y0_f = y0_diab_f
    # --- THIS BLOCK IS NOW CORRECTLY INDENTED ---
    if qsys.is_adiabatic:
        x0_d, p0_d, R0, P0 = unflatten(y0_diab_f, qsys.F, qsys.n_modes)
        _, U0 = qsys.nuclear_model.get_adiabatic_properties(R0, np.identity(qsys.F))
        x0_a = U0.T.conj() @ x0_d
        p0_a = U0.T.conj() @ p0_d
        y0_f = np.concatenate([R0.flatten(), P0.flatten(), x0_a, p0_a])

    end_t_au = params["end_time_au"]
    n_t_out = int(params.get("n_times", 100))
    t_eval_points = np.linspace(0, end_t_au, n_t_out)

    try:
        sol = solve_ivp(
            qsys.derivs, (0, end_t_au), y0_f,
            method='DOP853', t_eval=t_eval_points,
            atol=params.get("ode_atol", 1e-8), rtol=params.get("ode_rtol", 1e-8)
        )

        if sol.success:
            path_x_prop, path_p_prop, path_R_prop, _ = unflatten_solution_array_all_times(sol.y.T, qsys.F, qsys.n_modes)
            pops = np.zeros((n_t_out, qsys.F))
            U_prev = np.identity(qsys.F)

            for k_t in range(n_t_out):
                if qsys.is_adiabatic:
                    _, U = qsys.nuclear_model.get_adiabatic_properties(path_R_prop[k_t], U_prev)
                    U_prev = U
                    x_dia = U @ path_x_prop[k_t, :]
                    p_dia = U @ path_p_prop[k_t, :]
                    actions, _ = nq_from_xp(x_dia, p_dia, qsys.L)
                else:
                    actions, _ = nq_from_xp(path_x_prop[k_t, :], path_p_prop[k_t, :], qsys.L)

                pops[k_t, :] = get_histogram_population(actions, qsys.L, qsys.F)

            return TrajectoryData(
                raw_diabatic_pops_vs_time=pops,
                raw_adiabatic_pops_vs_time=np.array([]),
                E_total_vs_time=np.array([]),
                is_bad_trajectory=False,
                original_trajectory_index=global_traj_idx
            )
    except Exception as e:
        print(f"Warning: Trajectory {global_traj_idx} failed with error: {e}", file=sys.stderr)

    return TrajectoryData(np.array([]), np.array([]), np.array([]), True, global_traj_idx)
