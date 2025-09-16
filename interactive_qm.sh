#!/bin/bash
#=============================================================================
#   Interactive Run Script for Direct Dynamics QM Model with TCPB (Serial)
#=============================================================================
set -euo pipefail

# --- Configuration ---
TOTAL_TRAJECTORIES=1
# --- FIX: Define the output path for the .argm file ---
BASE_ARGM_FILE="rst/params_qm.argm"
# --- END FIX ---
PYTHON_SCRIPT="main.py"
# ======================================================================
# --- MODIFICATION: Set the number of electronic states (F) ---
# ======================================================================
F_STATES=2
# ======================================================================

# --- TCPB Configuration ---
TCPB_HOSTNAME="localhost"
TCPB_PORT=12345

fuser -k ${TCPB_PORT}/tcp 2>/dev/null || true

# --- Start TeraChem Server ---
echo ">>> Starting TeraChem server..."
eval "$(conda shell.bash hook)"
conda activate py36a100
module --ignore-cache load gcc/9.3.0
module --ignore-cache load protobuf/3.11.2
module --ignore-cache load libmatheval/1.1.11

set +u
source /home/rliang/intel/parallel_studio_xe_2019.5.075/bin/psxevars.sh
set -u

export TeraChem=/lustre/work/rliang/terachem-tip-a100/production/build
export LD_LIBRARY_PATH=$TeraChem/lib:/home/pan60047/anaconda3/envs/py36a100/lib:$LD_LIBRARY_PATH
module unload cuda
module load cuda
export OpenMM=/home/pan60047/anaconda3/envs/py36a100
export OPENMM_PLUGIN_DIR=$OpenMM/lib/plugins

$TeraChem/bin/terachem -s ${TCPB_PORT} &
SERVER_PID=$!
echo ">>> TeraChem server started with PID: ${SERVER_PID}"
echo ""
sleep 10

# --- Create Argument File ---
echo ">>> Creating argument file: ${BASE_ARGM_FILE}"
cat > "${BASE_ARGM_FILE}" << EOF
# Argument file for QM model run (TCPB backend)
output_file_prefix      qm-run
random_seed             1234
F                       ${F_STATES}
init_state              0
n_atoms                 1212
n_qm_atoms              42

# --- TCPB Server Configuration ---
tcpb_hostname           ${TCPB_HOSTNAME}
tcpb_port               ${TCPB_PORT}

# --- Path Configuration ---
prmtop_file             rst/system.prmtop
rst7_file               rst/system.rst7
qmregion_file           rst/system.qmregion
terachem_input_file     rst/terachem.inp

# --- Simulation Parameters ---
time_units              femtoseconds
end_time                0.2
n_times                 3
dt                      0.1
n_trajs                 ${TOTAL_TRAJECTORIES}
EOF

echo ">>> Argument file created."
echo ""

# --- Environment & Execution ---
echo ">>> Setting up environment for Python script..."
conda activate qmmm_env

echo ">>> Starting serial Python run..."
python3 "${PYTHON_SCRIPT}" "${BASE_ARGM_FILE}"

# --- Cleanup ---
echo ">>> Killing TeraChem server..."
kill $SERVER_PID 2>/dev/null || true
echo ">>> Done."
