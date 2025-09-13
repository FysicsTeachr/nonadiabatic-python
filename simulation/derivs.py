# simulation/derivs.py
import numpy as np

def derivs_meyer_miller_adiabatic(adiab_E, nac_tensor, x, p, R_dot_flat):
    F = len(x)
    dx_dt, dp_dt = np.zeros(F), np.zeros(F)
    
    dx_dt = adiab_E * p
    dp_dt = -adiab_E * x
    
    for i in range(F):
        for j in range(F):
            if i == j: continue
            nac_term = np.dot(nac_tensor[i, j, :], R_dot_flat)
            dx_dt[i] -= nac_term * x[j] * 0.5
            dp_dt[i] += nac_term * p[j] * 0.5
            
    return dx_dt, dp_dt

def get_nuclear_derivatives(x, p, P, Hel_dR_adia_list, nuclear_model, L, adiab_E, nac_tensor):
    """
    Calculates the time derivatives for the nuclear variables (R, P)
    using a unified approach for any number of electronic states (F).
    """
    F = len(x)
    n_nucl_coords = nuclear_model.n_atoms * 3

    dR_dt = P / nuclear_model.masses[:, np.newaxis]

    # --- UNIFIED FORCE CALCULATION ---
    n = 0.5 * (x**2 + p**2) - L
    mod = (1.0 - np.sum(n)) / F if F > 0 else 0.0
    
    C = np.zeros((F, F))
    for i in range(F):
        C[i, i] = n[i] + mod
        for j in range(i + 1, F):
            C[i, j] = C[j, i] = 0.5 * (x[i] * x[j] + p[i] * p[j])

    # 1. Hellmann-Feynman force: Calculated via einsum. This is now safe for F=1
    #    because Hel_dR_adia_list has the correct 3D shape.
    hf_force_flat = np.einsum('ij,kji->k', C, np.array(Hel_dR_adia_list)).real

    # 2. Non-adiabatic force correction (this term is zero when F=1)
    nac_force_flat = np.zeros_like(hf_force_flat)
    if F > 1:
        energy_diff = adiab_E[:, np.newaxis] - adiab_E
        for k in range(n_nucl_coords):
            dH_dR_off_diag_k = nac_tensor[:, :, k] * energy_diff
            nac_force_flat[k] = -np.einsum('ij,ji->', C, dH_dR_off_diag_k).real
    
    total_force_flat = hf_force_flat + nac_force_flat
    # --- END UNIFIED CALCULATION ---

    dP_dt = total_force_flat.reshape(nuclear_model.n_atoms, 3)
    
    return dR_dt, dP_dt
