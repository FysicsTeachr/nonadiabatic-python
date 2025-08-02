import numpy as np
import math
from typing import Tuple

# NOTE: This import will need to be updated where this file is called from.
# e.g., from utils.transformations import xp_from_nq
from utils.transformations import xp_from_nq

def sample_histogram_window(init_state,F,L,rng):
    # Calling zeros() from numpy
    x_initial, p_initial = np.zeros(F), np.zeros(F)
    for i in range(F):
        if i == init_state:
            # Calling uniform() from numpy.random.Generator
            n = rng.uniform(1.0 - L,1.0 + L)
        else:
            # Calling uniform() from numpy.random.Generator
            n = rng.uniform(-L, L)
        q=rng.uniform(-math.pi, math.pi)
        # Calling local function xp_from_nq
        x_initial[i], p_initial[i] = xp_from_nq(n,q,L)
    return x_initial, p_initial

def get_histogram_population(actions, L, F):
    indicators = np.zeros(F, dtype=float)
    for i in range(F):
        action_i = actions[i]
        is_s_occupied = (1.0 - L) <= action_i <= (1.0 + L)
        if not is_s_occupied:
            continue

        product_of_others = 1.0
        for j in range(F):
            if i == j:
                continue
            action_j = actions[j]
            is_j_unoccupied = (-L) <= action_j <= (L)
            if not is_j_unoccupied:
                product_of_others = 0.0
                break
        indicators[i] = product_of_others
    return indicators

def sample_triangular_window(
    initial_active_state: int,
    F: int,
    L: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes electronic variables (x,p) using triangular windowing
    based on simple rejection sampling.
    """
    # Calling zeros() from numpy
    x_initial, p_initial = np.zeros(F), np.zeros(F)
    sampled_actions = np.zeros(F)
    # L variable already used internally with correct value
    if F != 2:
        raise NotImplementedError("Triangular window sampling is only "
                                  "implemented for 2 states.")
    s_idx, other_idx = initial_active_state, 1 - initial_active_state
    n_s_min, n_s_max = 1.0 - L, 2.0 - L
    n_other_min, n_other_max = -L, 1.0 - L
    while True:
        # Calling uniform() from numpy.random.Generator
        n_s_sampled = rng.uniform(n_s_min, n_s_max)
        # Calling uniform() from numpy.random.Generator
        n_other_sampled = rng.uniform(n_other_min, n_other_max)
        if n_s_sampled + n_other_sampled <= 2.0 - 2.0 * L:
            sampled_actions[s_idx] = n_s_sampled
            sampled_actions[other_idx] = n_other_sampled
            break
    for i in range(F):
        action_ni = sampled_actions[i]
        # Calling uniform() from numpy.random.Generator
        random_angle_qi = rng.uniform(-math.pi, math.pi)
        # Calling local function xp_from_nq
        x_initial[i], p_initial[i] = xp_from_nq(action_ni,
                                                 random_angle_qi, L)
    return x_initial, p_initial

def get_triangular_population(
    actions: np.ndarray,
    F: int
) -> np.ndarray:
    """
    Determines population indicators based on triangular windowing rules.
    """
    # For triangular window, L is uniquely determined to be 1/3
    L = 1.0 / 3.0
    indicators = np.zeros(F, dtype=float)
    if F != 2: # This logic is specifically for 2-state systems
        return indicators
    # Calling np.sum() from numpy
    if np.sum(actions) > (2.0 - 2.0 * L):
        return indicators
    for s_idx in range(F):
        is_in_window = True
        for j_idx in range(F):
            n_j = actions[j_idx]
            if s_idx == j_idx:
                if not (n_j >= 1.0 - L):
                    is_in_window = False
                    break
            else:
                if not (n_j >= -L):
                    is_in_window = False
                    break
        if is_in_window:
            indicators[s_idx] = 1.0
    # Windows are non-overlapping for triangular
    return indicators
