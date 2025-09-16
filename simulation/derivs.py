# simulation/derivs.py
import numpy as np

def derivs_meyer_miller_adiabatic(adiab_E,vnac_tensor,vx,p, R_dot_flat):
    F = len(x)
    dx_dt, dp_dt = np.zeros(F), np.zeros(F)
    
    # Diagonal part of the electronic Hamiltonian
    dx_dt = adiab_E * p
    dp_dt = -adiab_E * x
    
    # Off-diagonal part (non-adiabatic coupling)
    for i in range(F):
        for j in range(F):
            if i == j: continue
            nac_term = np.dot(nac_tensor[i, j, :], R_dot_flat)
            dx_dt[i] -= 0.5 * nac_term * x[j]
            dp_dt[i] += 0.5 * nac_term * p[j]
            
    return dx_dt, dp_dt

def get_nuclear_derivatives_meyer_miller(x, p, P, Hel_dR_adia_list, nuclear_model, L):
    F = len(x)
    
    # Calculate nuclear velocity
    dR_dt = P / nuclear_model.masses[:, np.newaxis]

    # --- Calculate the Meyer-Miller electronic density matrix ---
    n = 0.5 * (x**2 + p**2) - L
    mod = (1.0 - np.sum(n)) / F if F > 0 else 0.0
    
    rho_adia = np.zeros((F, F))
    for i in range(F):
        rho_adia[i, i] = n[i] + mod
        for j in range(i + 1, F):
            rho_adia[i, j] = rho_adia[j, i] = 0.5 * (x[i] * x[j] + p[i] * p[j])

    # --- Calculate the force on the nuclei ---
    # The force is the expectation value of the gradient operator over the electronic state
    force_flat = -np.einsum('ij,kji->k', rho_adia, np.array(Hel_dR_adia_list)).real
    dP_dt = force_flat.reshape(nuclear_model.n_atoms, 3)
    
    return dR_dt, dP_dt
