#!/bin/bash

#=============================================================================
#
#   Interactive MPI Run Script for FMO Model
#
#=============================================================================

set -euo pipefail

# --- Configuration ---
NTASKS=${SLURM_NTASKS:-16}
TOTAL_TRAJECTORIES=12800
BASE_ARGM_FILE="params_fmo.argm"
# UPDATED: Path to main.py is now in the current directory
PYTHON_SCRIPT="main.py"
MPI_MODULE="impi/19.0.9"
CONDA_ENV="base"

# --- Create Argument File ---
echo ">>> Creating argument file: ${BASE_ARGM_FILE}"
cat > "${BASE_ARGM_FILE}" << EOF
# Argument file for FMO model run
output_file_prefix      fmo-run
random_seed             42
nuclear_model           fmo
window_model            histogram
F                       7
init_state              0
L                       0.366
# UPDATED: Path to Hel-7.txt is now in the current directory
Hel_file                Hel-7.txt
lambda_cm               35.0
gamma_fs                50.0
n_modes                 100
bath_mass               1.0
wmax_factor             7.0
bath_discretization     quadratic
T_kelvin                77.0
time_units              femtoseconds
end_time                1000.0
n_times                 1001
dt                      0.1
n_trajs                 ${TOTAL_TRAJECTORIES}
Diabatic                T
EOF

echo ">>> Argument file created."
echo ""


# --- Environment & Execution ---
echo ">>> Setting up environment..."
module load ${MPI_MODULE}
source activate ${CONDA_ENV}
echo ">>> Starting MPI run..."
ibrun -n ${NTASKS} python3 "${PYTHON_SCRIPT}" "${BASE_ARGM_FILE}"
conda deactivate
echo ">>> Done."
