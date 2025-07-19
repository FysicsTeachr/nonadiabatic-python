#!/bin/bash

#=============================================================================
#
#  Interactive MPI Run Script for Slurm Cluster
#
#  - Assumes `mpi4py` has been rebuilt against the cluster's system MPI.
#  - To be run inside an interactive allocation, for example:
#      salloc -p nocona -N 2 -n 128 --time=02:00:00
#
#=============================================================================

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# --- Configuration ---
# Use the number of tasks from the Slurm allocation, or default to 128
NTASKS=${SLURM_NTASKS:-128}
# Set total trajectories; here we use 1 trajectory per MPI rank
TOTAL_TRAJECTORIES=8192 #8192
# Name of the argument file to be generated
BASE_ARGM_FILE="params_interactive.argm"
# The unified Python script for MPI execution
PYTHON_SCRIPT="main.py"
# The specific MPI module mpi4py was compiled against
MPI_MODULE="intel-mpi/2019.10.317"
# The Conda environment where the rebuilt mpi4py resides
CONDA_ENV="base"


# --- Create Argument File ---
echo ">>> Creating argument file: ${BASE_ARGM_FILE}"
# This heredoc creates the input file for the python script.
# We are setting up a DIABATIC run based on your previous attempt.
cat > "${BASE_ARGM_FILE}" << EOF
# Argument file for interactive run (${TOTAL_TRAJECTORIES} trajectories)
output_file_prefix      interactive-run

random_seed 42

electronic_model        Meyer-Miller
nuclear_model           spin-boson
window_model            triangular
init_state              0
system                  spin-boson-adia
spectral_density        Ohmic
bath_eq_shift           1
bath_mass               1
L                       0.333
n_threads               1
n_trajs                 ${TOTAL_TRAJECTORIES}
energy_units            atomic_units
alpha           0.1
wc              2.5
wmax_factor     7
n_modes         100
beta            5
delta           1
epsilon         1
time_units              atomic_units
end_time                10
n_times                 100
dt                      0.0001
# --- Select Dynamics Type ---
# Set Diabatic to T for a diabatic run
Adiabatic               F
Diabatic                T
# --------------------------
Bin_diabatic            T
Off-diag                F
RK4                     F
dt_nucl                 0.0001
EOF
echo ">>> Argument file created."
echo ""


# --- Environment Setup ---
echo ">>> Setting up environment..."
echo "    > Purging existing modules for a clean environment."
module purge
echo "    > Loading system MPI module: ${MPI_MODULE}"
module load ${MPI_MODULE}
echo "    > Activating Conda environment: ${CONDA_ENV}"
source activate ${CONDA_ENV}
echo ">>> Environment setup complete."
echo ""


# --- Execution ---
echo ">>> Starting MPI run with ${NTASKS} tasks..."
echo "    Each MPI rank will process $((TOTAL_TRAJECTORIES / NTASKS)) trajectory/trajectories."
echo "--------------------------------------------------------"

# Use srun to launch the Python script. It will now work correctly because
# the loaded Intel MPI libraries match what mpi4py is linked against.
##srun python3 "${PYTHON_SCRIPT}" "${BASE_ARGM_FILE}" --global_idx_offset 0
#srun --mpi=pmi2 python3 "${PYTHON_SCRIPT}" "${BASE_ARGM_FILE}" --global_idx_offset 0
srun -n ${NTASKS} --mpi=pmi2 python3 "${PYTHON_SCRIPT}" "${BASE_ARGM_FILE}" --global_idx_offset 0
echo "--------------------------------------------------------"
echo ">>> MPI run finished."


# --- Cleanup ---
echo ">>> Deactivating Conda environment."
conda deactivate
echo ">>> Done."
