# simulation/initialize_trajectory.py
import numpy as np

from models.window_options import sample_histogram_window
from models.qm import QM
from utils.transformations import flatten

# --- Factory for selecting the windowing function ---
WINDOW_DISPATCH = {
    "histogram": sample_histogram_window,
    # Add other window types like "triangular" here if needed
}

def initialize_traj(params, model, rng):
    init_state = int(params.get("init_state", 0))
    F = model.F
    # Note: L is not used by the QM model itself, but by the electronic sampler
    L = float(params.get("L", 0.366)) 

    # --- 1. Nuclear Initialization (from QM model) ---
    R, P = model.initialize_nuclear_coordinates(rng)

    # --- 2. Electronic Initialization (using windowing) ---
    window_name = params.get("window_model", "histogram").lower()
    if window_name not in WINDOW_DISPATCH:
        raise NotImplementedError(f"Window model '{window_name}' is not supported.")

    sampling_function = WINDOW_DISPATCH[window_name]
    x, p = sampling_function(init_state, F, L, rng)

    # --- 3. Flatten and Combine into a single state vector ---
    return flatten(x, p, R, P)
