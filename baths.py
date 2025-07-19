# baths.py
from typing import Dict, Any, Tuple
import numpy as np
import math

def build_spin_boson_bath_parameters(params):
    bath_N= int(params.get("n_modes", 100))
    m= params.get("bath_mass", 1.0)
    wc = params["wc"] 
    alpha_param = params["alpha"] 
    beta = params.get("beta", 1.0) 
    bath_eq_shift_val = params.get("bath_eq_shift", 1.0) 
    shift= np.array([bath_eq_shift_val, -bath_eq_shift_val])
    F = 2 
    wmax_factor = params.get("wmax_factor", 7.0) 
    w_max = wc * wmax_factor 
    omega_k = np.zeros(bath_N) 
    d_k = np.zeros(bath_N)

    for i in range(bath_N):
        omega_k[i] = ((i + 0.5) / bath_N)**2 * w_max
        dw_approx = (2.0 * (i + 0.5) / bath_N) * (w_max / bath_N) 
        valid_omega_i = max(omega_k[i], 1e-12) 

        J_omega_i = (math.pi / 2.0) * alpha_param * valid_omega_i * \
                    math.exp(-valid_omega_i / wc) 

        d_k_squared_numerator = (2.0 / math.pi) * J_omega_i * dw_approx 
        d_k_squared_denominator = m * (valid_omega_i**3) 

        d_k[i] = math.sqrt(d_k_squared_numerator / d_k_squared_denominator) \
                 if d_k_squared_denominator > 1e-24 else 0.0

    # Output: dictionary of built bath parameters
    return {
        "omega_k": omega_k,
        "d_k": d_k,
        "shift": shift,
        "bath_mass": m,
        "beta": beta,
        "F": F,
        "n_modes": bath_N 
    }


def sample_spin_boson_bath(built_bath_parameters,
    initial_diabatic_state_idx,rng):

    omega_k = built_bath_parameters["omega_k"] 
    d_k = built_bath_parameters["d_k"] 
    shift = built_bath_parameters["shift"] 
    m = built_bath_parameters["bath_mass"] 
    beta = built_bath_parameters["beta"] 
    bath_N = built_bath_parameters["n_modes"] 

    R_initial, P_initial = np.zeros(bath_N), np.zeros(bath_N) 

    for i in range(bath_N):
        omega_i = omega_k[i] 
        coth_term = 1.0 / math.tanh(beta * omega_i / 2.0) \
                    if beta > 0 and omega_i * beta > 1e-6 \
                    else (2.0 / (beta * omega_i) \
                          if beta > 0 and omega_i > 1e-9 else 1.0) 

        sigma_R_sq = (1.0 / (2.0*m*omega_i \
                              if omega_i > 1e-9 else 1e9)) * coth_term 
        sigma_P_sq = (m * (omega_i if omega_i > 1e-9 else 1e-9) / 2.0) \
                     * coth_term 

        R_i_unshifted = rng.normal(loc=0.0,
                                   scale=math.sqrt(max(sigma_R_sq, 1e-12)))
        P_initial[i] = rng.normal(loc=0.0,
                                  scale=math.sqrt(max(sigma_P_sq, 1e-12)))

        R_initial[i] = R_i_unshifted + \
                       d_k[i] * shift[initial_diabatic_state_idx]

    return R_initial, P_initial
