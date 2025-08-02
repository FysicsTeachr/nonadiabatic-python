#for larger number of trajectories > 32000, the previous main.py crashes. The following works; being tested for very large
import argparse
from pathlib import Path
import sys
from mpi4py import MPI
import numpy as np
import pickle ### NEW: Import the pickle module for saving/loading Python objects

# --- Updated Imports from the new modular structure ---
from utils.read_params import parse_argm
from models import baths
from simulation.run_single_trajectory import run_single_traj
from analysis.post_process import analyze_and_write_output
from analysis.data_types import TrajectoryData

def main() -> None:
    """
    Main driver for MPI-based non-adiabatic dynamics simulations.
    This script coordinates the simulation based on the refactored modules.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Argument Parsing (Root Rank Only) ---
    # This section remains unchanged
    args = None
    if rank == 0:
        parser = argparse.ArgumentParser(description="Python non-adiabatic dynamics simulation.")
        parser.add_argument("argm_file", type=Path, help="Path to the .argm input file.")
        parser.add_argument("--global_idx_offset", type=int, default=0)
        try:
            args = parser.parse_args()
        except SystemExit:
            args = None # Allows for graceful exit if args are invalid

    args = comm.bcast(args, root=0)
    if args is None:
        comm.Abort(1)
        sys.exit(1)

    # --- Parameter Setup ---
    # This section remains unchanged
    try:
        params = parse_argm(args.argm_file)
        params['global_idx_offset'] = args.global_idx_offset
        built_bath_params = baths.build_bath_parameters(params)
        params["built_bath_params"] = built_bath_params
    except Exception as e:
        print(f"Rank {rank}: Failed during setup. Error: {e}", file=sys.stderr)
        comm.Abort(1)
        sys.exit(1)

    # --- Trajectory Distribution Logic ---
    # This section remains unchanged
    total_trajs = int(params.get("n_trajs", 0))
    trajs_per_rank = total_trajs // size
    remainder = total_trajs % size
    local_n_trajs = trajs_per_rank + (1 if rank < remainder else 0)
    start_idx = rank * trajs_per_rank + min(rank, remainder)
    local_start_global_idx = params.get('global_idx_offset', 0) + start_idx

    # --- Simulation ---
    # This section remains unchanged
    local_results: list[TrajectoryData] = []
    if rank == 0:
        print(f"Master: Starting {total_trajs} trajectories across {size} ranks.")

    base_seed = params.get("random_seed", None)
    if base_seed is not None:
        rank_seed = int(base_seed) + local_start_global_idx
    else:
        rank_seed = local_start_global_idx
    rng = np.random.default_rng(rank_seed)

    for i in range(local_n_trajs):
        current_global_traj_idx = local_start_global_idx + i
        result = run_single_traj(params, current_global_traj_idx, rng)
        local_results.append(result)

    # --- Aggregation and Post-Processing ---
    ### NEW: SAVE-TO-DISK LOGIC STARTS HERE

    # 1. Define a directory for temporary results and create it on rank 0
    output_dir = Path("./temp_results")
    if rank == 0:
        output_dir.mkdir(exist_ok=True)
        print("Master: All ranks finished simulations. Writing temporary files...")

    # 2. Synchronize all processes to make sure the directory exists before writing
    comm.Barrier()

    # 3. Each rank saves its results to a unique file using pickle
    temp_file = output_dir / f"results_rank_{rank}.pkl"
    with open(temp_file, 'wb') as f:
        pickle.dump(local_results, f)

    # 4. Synchronize all processes to ensure all files are written before proceeding
    comm.Barrier()

    ### OLD LOGIC IS REPLACED. `comm.gather` IS GONE.

    # 5. Only Rank 0 will now aggregate and analyze
    if rank == 0:
        print("Master: All temporary files written. Aggregating and analyzing results...")

        # 6. Read all the individual files and combine them into one master list
        final_aggregated_data = []
        for i in range(size):
            rank_file = output_dir / f"results_rank_{i}.pkl"
            try:
                with open(rank_file, 'rb') as f:
                    rank_data = pickle.load(f)
                    final_aggregated_data.extend(rank_data)
                # Optional: Uncomment to clean up files as they are read
                # rank_file.unlink()
            except FileNotFoundError:
                print(f"Warning: Could not find results file for rank {i}", file=sys.stderr)

        # Optional: After aggregation, you can remove the temp directory
        # import shutil
        # shutil.rmtree(output_dir)

        # 7. Analysis proceeds as before on the fully aggregated data
        n_t_out = int(params.get("n_times", 100))
        end_t_au = params["end_time_au"]
        time_points_eval = np.linspace(0, end_t_au, n_t_out)

        final_output_prefix = params.get("output_file_prefix", "sim_output")
        params["output_file_prefix"] = f"final_aggregated_{final_output_prefix}"

        analyze_and_write_output(final_aggregated_data, params, time_points_eval)
        print("Master: Final analysis complete.")

if __name__ == "__main__":
    main()
