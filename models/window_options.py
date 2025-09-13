# models/window_options.py
import numpy as np
import math

from utils.transformations import xp_from_nq

def sample_histogram_window(init_state,F,L,rng):
    x_initial, p_initial = np.zeros(F), np.zeros(F)
    for i in range(F):
        if i == init_state:
            n = rng.uniform(1.0 - L,1.0 + L)
        else:
            n = rng.uniform(-L, L)
        q=rng.uniform(-math.pi, math.pi)
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
