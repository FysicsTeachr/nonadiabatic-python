# simulation/initialize_trajectory.py
import numpy as np

# --- Imports from the modular structure ---
from models.window_options import sample_histogram_window, sample_triangular_window
from models.base import NuclearModelBase
# NOTE: The 'flatten' utility is no longer needed here.

# --- Factory for selecting the windowing function ---
WINDOW_DISPATCH = {
    "histogram": sample_histogram_window,
    "triangular": sample_triangular_window,
}

def initialize_traj(params: dict, model: NuclearModelBase, rng: np.random.Generator) -> np.ndarray:
    """
    Initializes a single trajectory for a non-adiabatic simulation.
    """
    init_state = int(params.get("init_state", 0))
    F = model.F
    L = float(params.get("L", 0.366))

    # --- 1. Nuclear Initialization ---
    R, P = model.initialize_nuclear_coordinates(rng)

    # --- 2. Electronic Initialization ---
    window_name = params.get("window_model", "histogram").lower()
    if window_name not in WINDOW_DISPATCH:
        raise NotImplementedError(f"Window model '{window_name}' is not supported.")

    sampling_function = WINDOW_DISPATCH[window_name]
    x, p = sampling_function(init_state, F, L, rng)

    # --- 3. Flatten and Combine ---
    # CORRECTED: Use np.concatenate to assemble the initial state vector directly.
    # The `flatten` utility was incorrectly zeroing out the electronic parts (x, p).
    return np.concatenate([R.flatten(), P.flatten(), x, p])
