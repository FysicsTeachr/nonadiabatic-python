# main.py
import argparse
from pathlib import Path
import sys
import numpy as np
import pickle

# --- Imports from our new v2 structure ---
from utils.read_params import parse_argm
from simulation.run_single_trajectory_qm import run_single_traj_qm
from analysis.data_types import TrajectoryData
from analysis.post_process import analyze_and_write_output

def main():
    # --- Argument Parsing (Simplified version for a single process) ---
    parser = argparse.ArgumentParser(description="Python QM dynamics simulation.")
    parser.add_argument("argm_file", type=Path, help="Path to the .argm input file.")
    try:
        args = parser.parse_args()
    except SystemExit:
        print("Error: Invalid arguments provided.", file=sys.stderr)
        sys.exit(1)

    try:
        params = parse_argm(args.argm_file)
    except Exception as e:
        print(f"Failed during setup. Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Simulation (Serial Version) ---
    total_trajs = int(params.get("n_trajs", 1))
    print(f"Starting {total_trajs} trajectories in a single process.")

    base_seed = int(params.get("random_seed", 42))
    rng = np.random.default_rng(base_seed)

    all_results = []
    for i in range(total_trajs):
        result = run_single_traj_qm(params, i, rng)
        all_results.append(result)

    # --- Analysis (Simplified for a single process) ---
    print("Simulations complete. Starting final analysis...")
    
    # This ensures the analysis always matches the simulation length.
    if all_results and not all_results[0].is_bad_trajectory:
        n_t_out = len(all_results[0].E_total_vs_time)
    else:
        # Fallback if all trajectories failed
        n_t_out = int(params["end_time_au"] / params["dt_au"]) + 1
    
    end_t_au = params["end_time_au"]
    time_points_eval = np.linspace(0, end_t_au, n_t_out)

    analyze_and_write_output(all_results, params, time_points_eval)
    print("Final analysis complete.")


if __name__ == "__main__":
    main()
