
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List
from analysis import TrajectoryData # For TrajectoryData type hint

def process_and_print_populations(
    all_traj_data: List[TrajectoryData],
    params: Dict[str, Any],
    time_points_eval: np.ndarray
) -> None:
    """
    Processes and prints averaged diabatic state populations.
    """
    good_traj_data = [td for td in all_traj_data if not td.is_bad_trajectory]
    num_good_trajs = len(good_traj_data)

    if num_good_trajs == 0:
        # Calling print (inbuilt function)
        print("Warning: No good trajectories to print populations.")
        return

    # --- Averaging logic from analysis.py ---
    # Calling np.array (from numpy)
    # Calling np.sum (from numpy)
    summed_diab_pops = np.sum(np.array(
        [td.raw_diabatic_pops_vs_time for td in good_traj_data]), axis=0)
    # Calling np.sum (from numpy)
    norm_factor_diab = np.sum(summed_diab_pops, axis=1, keepdims=True)
    # Calling np.divide (from numpy)
    avg_diab_pops = np.divide(summed_diab_pops, norm_factor_diab,
                              out=np.zeros_like(summed_diab_pops),
                              where=norm_factor_diab > 0)
    # --- End averaging logic ---

    num_states = avg_diab_pops.shape[1]

    # Calling print (inbuilt function)
    print(f"\n--- Averaged Diabatic State Populations ({num_good_trajs} "
          f"good trajectories) ---")
    header_parts = ["Time"]
    # Calling inbuilt function range
    header_parts.extend([f"P{i}_dia_avg" for i in range(num_states)])
    # Calling join (inbuilt function)
    print("\t\t".join(header_parts))

    # Calling inbuilt function enumerate
    for k_t, t_val in enumerate(time_points_eval):
        line_parts = [f"{t_val:<12.6f}"]
        # Calling inbuilt function range
        line_parts.extend(
            [f"{avg_diab_pops[k_t, i_s]:<12.6f}" \
             for i_s in range(num_states)])
        # Calling join (inbuilt function)
        print("\t\t".join(line_parts))
    # Calling print (inbuilt function)
    print("----------------------------------------------------------")
