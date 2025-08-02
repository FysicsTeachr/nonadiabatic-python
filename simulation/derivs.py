# simulation/derivs.py
import numpy as np

def rho_elec(x, p, L, F):
    """
    Calculates the electronic density matrix from action-angle variables.
    This function is model-agnostic.
    """
    n = 0.5 * (x**2 + p**2) - L
    mod = (1.0 - np.sum(n)) / F if F > 0 else 0.0
    C = np.zeros((F, F))
    for i in range(F):
        C[i, i] = n[i] + mod
        for j in range(i + 1, F):
            C[i, j] = C[j, i] = 0.5 * (x[i] * x[j] + p[i] * p[j])
    return C

def derivs_meyer_miller_diabatic(H_diab, x, p, F):
    """
    Calculates the time derivatives for the electronic variables in the
    diabatic representation. This is part of a general propagation scheme.
    """
    dx_dt, dp_dt = np.zeros(F), np.zeros(F)
    for i in range(F):
        pi_dt_term, xi_dt_term = 0.0, 0.0
        for j in range(F):
            if i == j: continue
            hij = H_diab[i, j]
            h_dij = H_diab[i, i] - H_diab[j, j]
            pi_dt_term -= (hij * x[j] + h_dij / F * x[i])
            xi_dt_term += (hij * p[j] + h_dij / F * p[i])
        dp_dt[i], dx_dt[i] = pi_dt_term, xi_dt_term
    return dx_dt, dp_dt

def derivs_nuclear_adiabatic(model, rho_adia, Hel_dR_adia_list):
    """
    Calculates the nuclear force in the adiabatic representation.
    This is a general propagation scheme.
    """
    force_flat = -np.einsum('ij,kji->k', rho_adia, np.array(Hel_dR_adia_list)).real
    dP_dt = force_flat.reshape((model.F, model.n_modes))
    # In this specific implementation, dR_dt is calculated elsewhere, so we return zeros.
    # The actual dR_dt = P/m is handled in the main solver loop.
    dR_dt = np.zeros_like(dP_dt)
    return dR_dt, dP_dt

def derivs_mm_adia_2_site(adiab_E, nac_vectors_flat, x, p, R_dot_flat):
    """
    Adiabatic EOM for electronic variables for a 2-state system.
    This is part of the adiabatic propagation scheme.
    """
    F = len(x)
    dx_dt, dp_dt = np.zeros(F), np.zeros(F)
    for i in range(F):
        dx_dt[i] = adiab_E[i] * p[i]
        dp_dt[i] = -adiab_E[i] * x[i]
    nac_term = np.dot(nac_vectors_flat, R_dot_flat)
    dx_dt[0] += -nac_term * x[1]
    dp_dt[0] += -nac_term * p[1]
    dx_dt[1] += nac_term * x[0]
    dp_dt[1] += nac_term * p[0]
    return dx_dt, dp_dt

def derivs_mm_adia_general(adiab_E, nac_tensor, x, p, R_dot_flat):
    """
    Adiabatic EOM for electronic variables for a general F-state system.
    This is part of the adiabatic propagation scheme.
    """
    F = len(x)
    dx_dt, dp_dt = np.zeros(F), np.zeros(F)
    dx_dt = adiab_E * p
    dp_dt = -adiab_E * x
    for i in range(F):
        for j in range(F):
            if i == j: continue
            nac_term = np.dot(nac_tensor[i, j, :], R_dot_flat)
            dx_dt[i] -= 0.5 * nac_term * x[j]
            dp_dt[i] += 0.5 * nac_term * p[j]
    return dx_dt, dp_dt
