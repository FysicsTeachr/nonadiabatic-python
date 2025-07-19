# analysis.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class TrajectoryData:
    raw_diabatic_pops_vs_time: np.ndarray
    raw_adiabatic_pops_vs_time: np.ndarray
    E_total_vs_time: np.ndarray
    is_bad_trajectory: bool = False
    original_trajectory_index: int = -1

def analyze_and_write_output(
    all_traj_data: List[TrajectoryData],
    params: Dict[str, Any],
    time_points_eval: np.ndarray
) -> None:
    """
    Analyzes a list of TrajectoryData objects and writes the averaged results to a file.
    """
    output_prefix = params.get("output_file_prefix", "sim_output")
    output_dir_path_str = params.get('argm_file_path', '.')
    
    # --- Calling Path() from pathlib ---
    output_dir_path = Path(output_dir_path_str)
    
    # --- Calling parent attribute and / operator from pathlib.Path ---
    output_file_timedep = output_dir_path.parent / f"{output_prefix}_py_time_dep_results.txt"

    good_traj_data = [td for td in all_traj_data if not td.is_bad_trajectory]
    num_good_trajs = len(good_traj_data)

    if num_good_trajs == 0:
        print("Warning: No good trajectories to analyze.")
        return

    # --- Calling np.sum() from numpy ---
    summed_diab_pops = np.sum(np.array([td.raw_diabatic_pops_vs_time for td in good_traj_data]), axis=0)
    # --- Calling np.sum() from numpy ---
    summed_adia_pops = np.sum(np.array([td.raw_adiabatic_pops_vs_time for td in good_traj_data]), axis=0)
    # --- Calling np.mean() from numpy ---
    avg_E_tot = np.mean(np.array([td.E_total_vs_time for td in good_traj_data]), axis=0)

    # --- Calling np.sum() from numpy ---
    norm_factor_diab = np.sum(summed_diab_pops, axis=1, keepdims=True)
    # --- Calling np.sum() from numpy ---
    norm_factor_adia = np.sum(summed_adia_pops, axis=1, keepdims=True)
    # --- Calling np.divide() from numpy ---
    avg_diab_pops = np.divide(summed_diab_pops, norm_factor_diab, out=np.zeros_like(summed_diab_pops), where=norm_factor_diab > 0)
    # --- Calling np.divide() from numpy ---
    avg_adia_pops = np.divide(summed_adia_pops, norm_factor_adia, out=np.zeros_like(summed_adia_pops), where=norm_factor_adia > 0)
    
    num_states = avg_diab_pops.shape[1]

    # --- Calling open() built-in function ---
    with open(output_file_timedep, 'w') as f:
        f.write(f"# Time-Dependent Results (Python Simulation)\n")
        f.write(f"# Used {num_good_trajs} trajectories for analysis.\n")
        header_parts = ["#Time"]
        header_parts.extend([f"P{i}_adia_avg" for i in range(num_states)])
        header_parts.extend([f"P{i}_dia_avg" for i in range(num_states)])
        header_parts.append("Etot_avg")
        f.write("\t\t".join(header_parts) + "\n")

        for k_t, t_val in enumerate(time_points_eval):
            line_parts = [f"{t_val:<12.6f}"]
            line_parts.extend([f"{avg_adia_pops[k_t, i_s]:<12.6f}" for i_s in range(num_states)])
            line_parts.extend([f"{avg_diab_pops[k_t, i_s]:<12.6f}" for i_s in range(num_states)])
            line_parts.append(f"{avg_E_tot[k_t]:<12.6f}")
            f.write("\t\t".join(line_parts) + "\n")
    
    # --- Calling resolve() method from pathlib.Path ---
    print(f"Final time-dependent results written to {output_file_timedep.resolve()}")
