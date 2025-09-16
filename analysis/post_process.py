# analysis/post_process.py
from pathlib import Path
import numpy as np
from .data_types import TrajectoryData

def analyze_and_write_output(
    all_traj_data,
    params,
    time_points_eval
):
    """
    Analyzes a list of TrajectoryData objects and writes averaged results to a file.
    """
    output_prefix = params.get("output_file_prefix", "sim_output")
    output_file_timedep = Path(f"{output_prefix}_py_time_dep_results.txt")

    good_traj_data = [
        td for td in all_traj_data
        if not td.is_bad_trajectory and td.raw_adiabatic_pops_vs_time.size > 0
    ]
    num_good_trajs = len(good_traj_data)

    if num_good_trajs == 0:
        print("Warning: No good trajectories to analyze.")
        return

    # Process Adiabatic Populations
    summed_adiab_pops = np.sum(np.array([td.raw_adiabatic_pops_vs_time for td in good_traj_data]), axis=0)
    # Normalize populations at each timestep
    norm_factor = np.sum(summed_adiab_pops, axis=1, keepdims=True)
    avg_adiab_pops = np.divide(summed_adiab_pops, norm_factor, out=np.zeros_like(summed_adiab_pops), where=norm_factor > 0)
    num_states = avg_adiab_pops.shape[1]

    # Process Total Energy (and zero at t=0)
    zeroed_E_total_trajs = []
    for td in good_traj_data:
        zeroed_E_total_trajs.append(td.E_total_vs_time - td.E_total_vs_time[0])

    summed_E_total = np.sum(np.array(zeroed_E_total_trajs), axis=0)
    avg_E_total = summed_E_total / num_good_trajs


    with open(output_file_timedep, 'w') as f:
        f.write(f"# Time-Dependent Results (Python QM Simulation)\n")
        f.write(f"# Used {num_good_trajs} trajectories for analysis.\n")

        header_parts = ["#Time", "E_total"]
        header_parts.extend([f"P{i}_adia_avg" for i in range(num_states)])
        f.write("\t\t".join(header_parts) + "\n")

        for k_t, t_val in enumerate(time_points_eval):
            line_parts = [f"{t_val:<12.6f}", f"{avg_E_total[k_t]:<12.6f}"]
            line_parts.extend([f"{avg_adiab_pops[k_t, i_s]:<12.6f}" for i_s in range(num_states)])
            f.write("\t\t".join(line_parts) + "\n")

    print(f"Final time-dependent results written to {output_file_timedep.resolve()}")
