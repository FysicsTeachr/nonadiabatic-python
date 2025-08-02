# analysis/post_process.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

# Import the data structure from the sibling module
from .data_types import TrajectoryData

def analyze_and_write_output(
    all_traj_data: List[TrajectoryData],
    params: Dict[str, Any],
    time_points_eval: np.ndarray
) -> None:
    """
    Analyzes a list of TrajectoryData objects and writes the averaged results to a file.
    (Formerly the main function in analysis.py)
    """
    output_prefix = params.get("output_file_prefix", "sim_output")
    # This path logic assumes the script is run from the project root
    output_dir_path = Path(params.get('argm_file_path', '.')).parent
    output_file_timedep = output_dir_path / f"{output_prefix}_py_time_dep_results.txt"

    good_traj_data = [td for td in all_traj_data if not td.is_bad_trajectory]
    num_good_trajs = len(good_traj_data)

    if num_good_trajs == 0:
        print("Warning: No good trajectories to analyze.")
        return

    # --- Check if adiabatic and energy data exist before processing ---
    has_adia_data = (len(good_traj_data) > 0 and
                     good_traj_data[0].raw_adiabatic_pops_vs_time.size > 0)
    has_energy_data = (len(good_traj_data) > 0 and
                       good_traj_data[0].E_total_vs_time.size > 0)

    # --- Process Diabatic Populations (always expected) ---
    summed_diab_pops = np.sum(np.array([td.raw_diabatic_pops_vs_time for td in good_traj_data]), axis=0)
    norm_factor_diab = np.sum(summed_diab_pops, axis=1, keepdims=True)
    avg_diab_pops = np.divide(summed_diab_pops, norm_factor_diab, out=np.zeros_like(summed_diab_pops), where=norm_factor_diab > 0)
    num_states = avg_diab_pops.shape[1]

    # --- Conditionally Process Adiabatic Populations ---
    avg_adia_pops = None
    if has_adia_data:
        summed_adia_pops = np.sum(np.array([td.raw_adiabatic_pops_vs_time for td in good_traj_data]), axis=0)
        norm_factor_adia = np.sum(summed_adia_pops, axis=1, keepdims=True)
        avg_adia_pops = np.divide(summed_adia_pops, norm_factor_adia, out=np.zeros_like(summed_adia_pops), where=norm_factor_adia > 0)

    # --- Conditionally Process Energy ---
    avg_E_tot = None
    if has_energy_data:
        avg_E_tot = np.mean(np.array([td.E_total_vs_time for td in good_traj_data]), axis=0)

    # --- Write Output File ---
    with open(output_file_timedep, 'w') as f:
        f.write(f"# Time-Dependent Results (Python Simulation)\n")
        f.write(f"# Used {num_good_trajs} trajectories for analysis.\n")

        header_parts = ["#Time"]
        if has_adia_data:
            header_parts.extend([f"P{i}_adia_avg" for i in range(num_states)])
        header_parts.extend([f"P{i}_dia_avg" for i in range(num_states)])
        if has_energy_data:
            header_parts.append("Etot_avg")
        f.write("\t\t".join(header_parts) + "\n")

        for k_t, t_val in enumerate(time_points_eval):
            line_parts = [f"{t_val:<12.6f}"]
            if has_adia_data and avg_adia_pops is not None:
                line_parts.extend([f"{avg_adia_pops[k_t, i_s]:<12.6f}" for i_s in range(num_states)])
            line_parts.extend([f"{avg_diab_pops[k_t, i_s]:<12.6f}" for i_s in range(num_states)])
            if has_energy_data and avg_E_tot is not None:
                line_parts.append(f"{avg_E_tot[k_t]:<12.6f}")
            f.write("\t\t".join(line_parts) + "\n")

    print(f"Final time-dependent results written to {output_file_timedep.resolve()}")


def process_and_print_populations(
    all_traj_data: List[TrajectoryData],
    params: Dict[str, Any],
    time_points_eval: np.ndarray
) -> None:
    """
    Processes and prints averaged diabatic state populations to the console.
    (Formerly in process_output.py)
    """
    good_traj_data = [td for td in all_traj_data if not td.is_bad_trajectory]
    num_good_trajs = len(good_traj_data)

    if num_good_trajs == 0:
        print("Warning: No good trajectories to print populations.")
        return

    summed_diab_pops = np.sum(np.array(
        [td.raw_diabatic_pops_vs_time for td in good_traj_data]), axis=0)
    norm_factor_diab = np.sum(summed_diab_pops, axis=1, keepdims=True)
    avg_diab_pops = np.divide(summed_diab_pops, norm_factor_diab,
                              out=np.zeros_like(summed_diab_pops),
                              where=norm_factor_diab > 0)

    num_states = avg_diab_pops.shape[1]

    print(f"\n--- Averaged Diabatic State Populations ({num_good_trajs} "
          f"good trajectories) ---")
    header_parts = ["Time"]
    header_parts.extend([f"P{i}_dia_avg" for i in range(num_states)])
    print("\t\t".join(header_parts))

    for k_t, t_val in enumerate(time_points_eval):
        line_parts = [f"{t_val:<12.6f}"]
        line_parts.extend(
            [f"{avg_diab_pops[k_t, i_s]:<12.6f}" \
             for i_s in range(num_states)])
        print("\t\t".join(line_parts))
    print("----------------------------------------------------------")
