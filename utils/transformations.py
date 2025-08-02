# utils/transformations.py
import numpy as np
import math

def xp_from_nq(n, q, L):
    n_eff = n + L
    if n_eff <= 0: return 0.0, 0.0
    x = math.sqrt(2.0 * n_eff) * math.cos(q)
    p = -math.sqrt(2.0 * n_eff) * math.sin(q)
    return x, p

def nq_from_xp(x, p, L):
    n = 0.5 * (x**2 + p**2) - L
    q = np.arctan2(-p, x)
    return n, q

def unflatten(state_vector, F, n_modes):
    n_nucl_coords = F * n_modes
    idx_R_end = n_nucl_coords
    idx_P_end = 2 * n_nucl_coords
    idx_x_end = 2 * n_nucl_coords + F

    R_flat = state_vector[0:idx_R_end]
    P_flat = state_vector[idx_R_end:idx_P_end]
    
    extracted_R = R_flat.reshape((F, n_modes))
    extracted_P = P_flat.reshape((F, n_modes))
    extracted_x = state_vector[idx_P_end:idx_x_end]
    extracted_p = state_vector[idx_x_end:]
    
    return extracted_x, extracted_p, extracted_R, extracted_P

def flatten(dx_dt_elec, dp_dt_elec, dR_dt_nucl, dP_dt_nucl):
    dR_flat = dR_dt_nucl.flatten()
    dP_flat = dP_dt_nucl.flatten()
    return np.concatenate([dR_flat, dP_flat, dx_dt_elec, dp_dt_elec])

def unflatten_solution_array_all_times(sol_y_T, F, n_modes):
    n_nucl_coords = F * n_modes
    idx_R_end = n_nucl_coords
    idx_P_end = 2 * n_nucl_coords
    idx_x_end = 2 * n_nucl_coords + F

    R_flat_t = sol_y_T[:, 0:idx_R_end]
    P_flat_t = sol_y_T[:, idx_R_end:idx_P_end]
    
    path_R = R_flat_t.reshape((-1, F, n_modes))
    path_P = P_flat_t.reshape((-1, F, n_modes))
    path_x = sol_y_T[:, idx_P_end:idx_x_end]
    path_p = sol_y_T[:, idx_x_end:]
    
    return (path_x, path_p, path_R, path_P)
