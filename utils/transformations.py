# utils/transformations.py
import numpy as np
import math

def xp_from_nq(n, q, L):
    """Converts action-angle (n,q) to Cartesian (x,p) variables."""
    n_eff = n + L
    if n_eff <= 0: return 0.0, 0.0
    x = math.sqrt(2.0 * n_eff) * math.cos(q)
    p = -math.sqrt(2.0 * n_eff) * math.sin(q)
    return x, p

def nq_from_xp(x, p, L):
    """Converts Cartesian (x,p) to action-angle (n,q) variables."""
    x_arr = np.atleast_1d(x)
    p_arr = np.atleast_1d(p)
    n = 0.5 * (x_arr**2 + p_arr**2) - L
    q = np.arctan2(-p_arr, x_arr)
    return (n[0], q[0]) if np.isscalar(x) else (n, q)

def unflatten(state_vector, F, n_atoms):
    """
    Un-flattens the state vector.
    Correct Order: [x, p, R_flat, P_flat]
    """
    idx_x_end = F
    idx_p_end = 2 * F
    idx_R_end = 2 * F + n_atoms * 3

    extracted_x = state_vector[0:idx_x_end]
    extracted_p = state_vector[idx_x_end:idx_p_end]
    R_flat = state_vector[idx_p_end:idx_R_end]
    P_flat = state_vector[idx_R_end:]

    extracted_R = R_flat.reshape((n_atoms, 3))
    extracted_P = P_flat.reshape((n_atoms, 3))

    return extracted_x, extracted_p, extracted_R, extracted_P

def flatten(x, p, R, P):
    """
    Flattens the state vector and its derivatives.
    Correct Order: [x, p, R_flat, P_flat]
    """
    R_flat = R.flatten()
    P_flat = P.flatten()
    return np.concatenate([x, p, R_flat, P_flat])

def unflatten_solution_array_all_times(sol_y_T, F, n_atoms):
    """
    Un-flattens the entire trajectory solution array.
    """
    idx_x_end = F
    idx_p_end = 2 * F
    idx_R_end = 2 * F + n_atoms * 3

    path_x = sol_y_T[:, 0:idx_x_end]
    path_p = sol_y_T[:, idx_x_end:idx_p_end]
    R_flat_t = sol_y_T[:, idx_p_end:idx_R_end]
    P_flat_t = sol_y_T[:, idx_R_end:]

    path_R = R_flat_t.reshape((-1, n_atoms, 3))
    path_P = P_flat_t.reshape((-1, n_atoms, 3))

    return (path_x, path_p, path_R, path_P)
