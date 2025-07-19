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
    # Now expecting state_vector to be in the order [R, P, x, p]
    idx_R_end = n_modes
    idx_P_end = 2 * n_modes
    idx_x_end = 2 * n_modes + F

    extracted_R = state_vector[0:idx_R_end]
    extracted_P = state_vector[idx_R_end:idx_P_end]
    extracted_x = state_vector[idx_P_end:idx_x_end]
    extracted_p = state_vector[idx_x_end:]
    
    # Return in the order (x, p, R, P) as expected by the caller (SystemForSolver.derivs)
    return extracted_x, extracted_p, extracted_R, extracted_P

def flatten( dx_dt_elec, dp_dt_elec, dR_dt_nucl, dP_dt_nucl ):
    # Concatenate in the order [dR_dt_nucl, dP_dt_nucl, dx_dt_elec, dp_dt_elec]
    # to match the [R, P, x, p] order of the state vector and derivatives for the ODE solver
    d_trajectory_flat_dt = np.concatenate([
        dR_dt_nucl, dP_dt_nucl, dx_dt_elec, dp_dt_elec ])
    return d_trajectory_flat_dt

def unflatten_solution_array_all_times(sol_y_T, F, n_modes ):
    # Now expecting sol_y_T to be in the order [R, P, x, p] across time
    # Calculate slice indices based on this new order
    idx_R_end = n_modes
    idx_P_end = 2 * n_modes
    idx_x_end = 2 * n_modes + F

    path_R = sol_y_T[:, 0:idx_R_end]
    path_P = sol_y_T[:, idx_R_end:idx_P_end]
    path_x = sol_y_T[:, idx_P_end:idx_x_end]
    path_p = sol_y_T[:, idx_x_end:]
    
    # Return in the order (x, p, R, P) as expected by the caller for analysis
    return (path_x, path_p, path_R, path_P)
