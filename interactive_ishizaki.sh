#!/bin/bash

#=============================================================================
#
#   Interactive MPI Run Script for the 7-Site FMO Model
#
#=============================================================================

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# --- Configuration ---

NTASKS=${SLURM_NTASKS:-128}
TOTAL_TRAJECTORIES=8192 #8192
BASE_ARGM_FILE="params_fmo.argm"
PYTHON_SCRIPT="main.py"
MPI_MODULE="intel-mpi/2019.10.317"
CONDA_ENV="base"



# --- Create Argument File ---
echo ">>> Creating argument file: ${BASE_ARGM_FILE}"
cat > "${BASE_ARGM_FILE}" << EOF
# Argument file for Ishizaki model run
output_file_prefix      ishizaki-run
random_seed             42
nuclear_model           ishizaki
window_model            histogram
F                       2
init_state              0
L                       0.366
# UPDATED: Path to Hel-2.txt is now in the current directory
Hel_file                Hel-2.txt
lambda_cm               2.0
gamma_cm                53.08
n_modes                 100
bath_mass               1.0
wmax_factor             7.0
bath_discretization     linear
T_kelvin                300.0
time_units              femtoseconds
end_time                100.0
n_times                 101
dt                      0.1
n_trajs                 ${TOTAL_TRAJECTORIES}
Adiabatic               T
EOF
echo ">>> Argument file created."
echo ""


# --- Environment & Execution ---
echo ">>> Setting up environment..."
module load ${MPI_MODULE}
source activate ${CONDA_ENV}
echo ">>> Starting MPI run..."
srun -n ${NTASKS} --mpi=pmi2 python3 "${PYTHON_SCRIPT}" "${BASE_ARGM_FILE}" --global_idx_offset 0
conda deactivate
echo ">>> Done."
