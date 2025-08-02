# models/baths.py
import numpy as np

def build_bath_parameters(params: dict) -> dict:
    """
    Builds bath parameters from a Debye spectral density using either
    linear or quadratic discretization.
    """
    n_modes = int(params.get("n_modes", 100))
    m = float(params.get("bath_mass", 1.0))
    F = int(params.get("F", 2))
    discretization_method = params.get("bath_discretization", "linear").lower()

    lmbda = params["lambda"]
    gamma = params["gamma"] # wc
    beta = params.get("beta", 1.0)
    wmax_factor = float(params.get("wmax_factor", 7.0))

    w_max = gamma * wmax_factor
    omega_k = np.zeros(n_modes)
    c_k = np.zeros(n_modes)
    d_k = np.zeros(n_modes)

    if discretization_method == "quadratic":
        for i in range(n_modes):
            omega_k[i] = w_max * ((i + 0.5) / n_modes)**2
            dw_approx = 2.0 * (i + 0.5) / n_modes * (w_max / n_modes)
            J_omega_i = 2 * lmbda * omega_k[i] * gamma / (omega_k[i]**2 + gamma**2)
            c_k_squared = (2 * m * omega_k[i] / np.pi) * J_omega_i * dw_approx
            c_k[i] = np.sqrt(max(c_k_squared, 0))
    elif discretization_method == "linear":
        dw = w_max / n_modes
        for i in range(n_modes):
            omega_k[i] = (i + 1) * dw
            J_omega_i = 2 * lmbda * omega_k[i] * gamma / (omega_k[i]**2 + gamma**2)
            c_k_squared = (2 * m * omega_k[i] / np.pi) * J_omega_i * dw
            c_k[i] = np.sqrt(max(c_k_squared, 0))
    else:
        raise NotImplementedError(f"Bath discretization '{discretization_method}' is not supported.")

    for i in range(n_modes):
        if m > 1e-9 and omega_k[i] > 1e-9:
            d_k[i] = c_k[i] / (m * omega_k[i]**2)
        else:
            d_k[i] = 0.0

    return {
        "omega_k": omega_k, "c_k": c_k, "d_k": d_k,
        "bath_mass": m, "beta": beta, "n_modes": n_modes, "F": F
    }


def sample_bath_wigner(built_bath_params: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Samples initial nuclear coordinates and momenta from Wigner distribution."""
    p = built_bath_params
    R_initial = np.zeros((p["F"], p["n_modes"]))
    P_initial = np.zeros((p["F"], p["n_modes"]))

    for i_pigment in range(p["F"]):
        for k_mode in range(p["n_modes"]):
            omega_i = p["omega_k"][k_mode]
            coth_term = 1.0 / np.tanh(p["beta"] * omega_i / 2.0) if p["beta"] * omega_i > 1e-6 else 2.0 / (p["beta"] * omega_i)
            sigma_R_sq = (1.0 / (2.0 * p["bath_mass"] * omega_i)) * coth_term
            sigma_P_sq = (p["bath_mass"] * omega_i / 2.0) * coth_term
            R_initial[i_pigment, k_mode] = rng.normal(loc=0.0, scale=np.sqrt(max(sigma_R_sq, 1e-12)))
            P_initial[i_pigment, k_mode] = rng.normal(loc=0.0, scale=np.sqrt(max(sigma_P_sq, 1e-12)))
    return R_initial, P_initial

def shift_bath_coordinates(R: np.ndarray, params: dict, built_bath_params: dict) -> np.ndarray:
    """Explicitly shifts the bath coordinates for the initial electronic state."""
    init_state = int(params.get("init_state", 0))
    R_shifted = R.copy()
    R_shifted[init_state, :] += built_bath_params["d_k"]
    return R_shifted
