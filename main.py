import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List
from mpi4py import MPI
import numpy as np
from datetime import datetime

# --- Core simulation and analysis modules ---
from read_params import parse_argm
import baths
from run_single_trajectory import run_single_traj
from analysis import analyze_and_write_output, TrajectoryData
from process_output import process_and_print_populations

def main() -> None:
    """
    Main execution function using a robust "All Read" setup to prevent deadlocks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Step 1: Argument Parsing (Rank 0) and Broadcast of ARGS ---
    # We only broadcast the small 'args' object, which is very safe.
    args = None
    if rank == 0:
        parser = argparse.ArgumentParser(description="Python non-adiabatic dynamics simulation (Robust MPI).")
        parser.add_argument("argm_file", type=Path, help="Path to the .argm input file.")
        parser.add_argument("--global_idx_offset", type=int, default=0, help="Starting index for trajectories in this MPI job.")
        try:
            args = parser.parse_args()
        except SystemExit:
            args = None

    args = comm.bcast(args, root=0)
    if args is None:
        comm.Abort(1)
        sys.exit(1)

    # --- Step 2: Independent Setup on ALL Ranks ---
    # Each process reads the parameter file. This avoids a large, complex broadcast.
    try:
        params = parse_argm(args.argm_file)
        params['global_idx_offset'] = args.global_idx_offset
        if params.get("nuclear_model", "").lower() == "spin-boson":
            built_bath_params = baths.build_spin_boson_bath_parameters(params)
            params["built_bath_params"] = built_bath_params
    except Exception as e:
        print(f"Rank {rank}: Failed during setup. Error: {e}", file=sys.stderr)
        comm.Abort(1)
        sys.exit(1)

    # --- Step 3: Static Work Division ---
    total_trajs = int(params.get("n_trajs", 0))
    trajs_per_rank = total_trajs // size
    rem = total_trajs % size

    if rank < rem:
        local_n_trajs = trajs_per_rank + 1
        start_idx = rank * local_n_trajs
    else:
        local_n_trajs = trajs_per_rank
        start_idx = rem * (trajs_per_rank + 1) + (rank - rem) * trajs_per_rank

    global_offset = params.get('global_idx_offset', 0)
    local_start_global_idx = global_offset + start_idx

    # --- Step 4: Local Trajectory Execution ---
    local_results: List[TrajectoryData] = []
    if rank == 0:
        print(f"Master (Rank 0): Starting {total_trajs} trajectories across {size} ranks.")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f"[{current_time}] Rank {rank}: Starting {local_n_trajs} trajectories from global index {local_start_global_idx}.")

    # Initialize RNG once per rank, based on the global starting index for this rank
    base_seed = params.get("random_seed", None)
    rank_seed = int(base_seed) + local_start_global_idx if base_seed is not None else local_start_global_idx
    rng = np.random.default_rng(rank_seed)

    for i_traj in range(local_n_trajs):
        current_global_traj_idx = local_start_global_idx + i_traj
        # Pass the rank-specific rng object to run_single_traj
        result = run_single_traj(params, current_global_traj_idx, rng)
        local_results.append(result)

    print(f"Rank {rank}: Finished all {local_n_trajs} assigned trajectories.")

    # --- Step 5: Gather Results ---
    all_gathered_results = comm.gather(local_results, root=0)

    # --- Step 6: Final Analysis (Rank 0 only) ---
    if rank == 0:
        print("--- Master (Rank 0): All ranks finished. Aggregating results... ---")
        final_aggregated_data = [item for sublist in all_gathered_results for item in sublist]

        n_t_out = int(params.get("n_times", 100))
        end_t = float(params.get("end_time", 1.0))
        time_points_eval = np.linspace(0, end_t, n_t_out)

        final_output_prefix = params.get("output_file_prefix", "sim_output")
        params["output_file_prefix"] = f"final_aggregated_{final_output_prefix}"

        analyze_and_write_output(final_aggregated_data, params, time_points_eval)
        process_and_print_populations(final_aggregated_data, params, time_points_eval)

        print("--- Master (Rank 0): Final analysis complete. ---")

if __name__ == "__main__":
    main()
