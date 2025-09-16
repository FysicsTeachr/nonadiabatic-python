# analysis/data_types.py
from dataclasses import dataclass
import numpy as np

@dataclass
class TrajectoryData:
    """
    A dataclass to hold the results from a single trajectory simulation.
    """
    raw_adiabatic_pops_vs_time: np.ndarray
    is_bad_trajectory: bool = False
    original_trajectory_index: int = -1
