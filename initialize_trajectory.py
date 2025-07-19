# New2/initialize_trajectory.py (Corrected Order)

from typing import Dict, Any
import numpy as np
from window_options import sample_histogram_window, sample_triangular_window
import baths

def initialize_traj(params,rng):
    F = int(params.get("F", 2))
    n_modes = int(params.get("n_modes", 100))

    trajectory_x = np.zeros(F)
    trajectory_p = np.zeros(F)
    trajectory_R = np.zeros(n_modes)
    trajectory_P = np.zeros(n_modes)

    init_state = int(params.get("init_state", 0))
    window_model = params.get("window_model", "histogram").lower()

    # --- THE FIX: The order of these blocks is critical ---

    # 1. Sample the NUCLEAR variables FIRST to match the original code.
    built_bath_params = params.get("built_bath_params", None)
    if built_bath_params and \
       params.get("nuclear_model", "").lower() == "spin-boson":
        trajectory_R, trajectory_P = baths.sample_spin_boson_bath(
                              built_bath_params, init_state, rng )

    # 2. Sample the ELECTRONIC variables SECOND.
    if window_model == "triangular":
        L = 1.0 / 3.0
    else:
        L = params.get("L", 0.366)
    
    if window_model == "histogram":
        trajectory_x, trajectory_p = sample_histogram_window(
                                      init_state, F, L, rng )
    elif window_model == "triangular":
        trajectory_x, trajectory_p = sample_triangular_window(
                                       init_state, F, L, rng )
    else:
        raise ValueError(
              f"Unknown window_model input: '{window_model}'")

    # Concatenate the final state vector in the original code's order: [R, P, x, p]
    trajectory = np.concatenate([trajectory_R, trajectory_P,
                                 trajectory_x, trajectory_p])
    return trajectory
