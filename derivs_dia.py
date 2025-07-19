# New2/derivs_dia.py (Final Corrected Version)

import numpy as np
import math
from typing import Dict, Any, List

from nucl_dia_H_and_dH import SpinBoson

def rho_elec(trajectory_x ,trajectory_p ,L ,F):
    actions_n = 0.5 * (trajectory_x**2 + trajectory_p**2) - L
    mod = (1.0 - np.sum(actions_n)) / F if F > 0 else 0.0
    C_matrix = np.zeros((F, F))
    for i in range(F):
        C_matrix[i, i] = actions_n[i] + mod
        for j in range(i + 1, F):
            C_matrix[i, j] = C_matrix[j, i] = 0.5 * \
                (trajectory_x[i] * trajectory_x[j] + \
                 trajectory_p[i] * trajectory_p[j])
    return C_matrix

# --- FINAL FIX ---
# This function's signature was changed to accept the pre-calculated Hamiltonian.
# The redundant internal calculation has been removed.
def derivs_meyer_miller(H_diab_elec: np.ndarray, trajectory_x: np.ndarray,
                         trajectory_p: np.ndarray, F: int):
    # This was the bug: H_diab_elec was recalculated here. It is now passed in.

    H11, H22 = H_diab_elec[0,0], H_diab_elec[1,1]
    H12 = H_diab_elec[0,1]

    H_diff_term = (H11 - H22) / 2.0
    dx_dt, dp_dt = np.zeros(F), np.zeros(F)

    if F == 2:
        dx_dt[0] = H_diff_term * trajectory_p[0] + H12 * trajectory_p[1]
        dp_dt[0] = -H_diff_term * trajectory_x[0] - H12 * trajectory_x[1]
        dx_dt[1] = -H_diff_term * trajectory_p[1] + H12 * trajectory_p[0]
        dp_dt[1] = H_diff_term * trajectory_x[1] - H12 * trajectory_x[0]
    else:
        raise NotImplementedError("derivs_meyer_miller is currently implemented for F=2 only.")
    return dx_dt, dp_dt

def derivs_nuclear(sb_model: SpinBoson,
                    rho_elec_matrix: np.ndarray,
                    trajectory_R: np.ndarray, trajectory_P: np.ndarray,
                    m: float, n_modes: int):
    
    dR_dt = np.zeros(n_modes)
    dP_dt = np.zeros(n_modes)

    for k in range(n_modes):
        dR_dt[k] = trajectory_P[k] / m
        dH_dRk_matrix = sb_model.dH_dRk(trajectory_R, k)
        force_k = -np.trace(rho_elec_matrix @ dH_dRk_matrix)
        dP_dt[k] = force_k
    return dR_dt, dP_dt
