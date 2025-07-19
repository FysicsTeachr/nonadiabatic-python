import numpy as np
import math

def xp_from_nq(n, q, L):
    n_eff = n + L
    if n_eff <= 0:
        return 0.0, 0.0
    x = math.sqrt(2.0 * n_eff) * math.cos(q)
    p = -math.sqrt(2.0 * n_eff) * math.sin(q)
    return x, p

def nq_from_xp(trajectory_x,trajectory_p,L):
    actions_n = 0.5 * (trajectory_x**2 + trajectory_p**2) - L
    angles_q = np.arctan2(-trajectory_p, trajectory_x)
    return actions_n, angles_q

def unflatten( state_vector, F, n_modes ):
    trajectory_x = state_vector[0:F]
    trajectory_p = state_vector[F:2*F]
    trajectory_R = state_vector[2*F:2*F + n_modes]
    trajectory_P = state_vector[2*F + n_modes:2*F + 2*n_modes]
    return trajectory_x, trajectory_p, trajectory_R, trajectory_P

def flatten( dx_dt_elec, dp_dt_elec, dR_dt_nucl, dP_dt_nucl ):
    # Calling concatenate from numpy
    d_trajectory_flat_dt = np.concatenate([
        dx_dt_elec, dp_dt_elec, dR_dt_nucl, dP_dt_nucl ])
    return d_trajectory_flat_dt

def unflatten_solution_array_all_times(sol_y_T, F, n_modes ):
    # Calling array from numpy
    path_x = sol_y_T[:, 0:F] # Renamed
    path_p = sol_y_T[:, F:2*F] # Renamed
    path_R = sol_y_T[:, 2*F:2*F + n_modes] # Renamed
    path_P = sol_y_T[:, 2*F + n_modes:2*F + 2*n_modes] # Renamed
    return (path_x, path_p, path_R, path_P)