This code uses my current conda environments py36a100 and qmmm_env. 
(Ask for access or install parmed and the dependencies for pytcpb and terachem)

This code also requires my pytcpb update src files uploaded in the current repository. Do this before use:
 Install pytcpb with Amber the standard way, and then 
replace the src/tcpb-cpp/src files with my uploaded ones, and recompile

Inputs: system.rst7, system.prmtop, system.qmregion. All in the rst folder. 

Can be submitted to Toreador partition or run in a Toreador interactive node 
for running in interactive: ./interactive_qm.sh
for submit: copy a toreador submission script into the folder and add ./interactive_qm.sh to it



Testing:
If you run the code as-is, you should see the following printed out:

CommBoxLoop: Server started, listening for jobs...
>>> Creating argument file: rst/params_qm.argm
>>> Argument file created.

>>> Setting up environment for Python script...
>>> Starting serial Python run...
Starting 1 trajectories in a single process.
QM Model Initialized: Connected to TCPB server at localhost:12345
Creating a dedicated MM-only system for OpenMM...
OpenMM context for pure MM forces created successfully.
Generating TeraChem input files in rst/ directory...
Input files generated.

STARTING TRAJECTORY 0
--- DEBUG (models/qm.py): Data sent to TeraChem at step 0 ---
QM Coords (first 2 atoms, Bohr):
[[-0.81447191  6.54223141 -4.27078076]
 [ 0.23432602  7.31701907 -5.85437115]]
MM Coords (first 2 atoms, Bohr):
[[-6.338141   -0.33637123  5.29501225]
 [-4.43518692 -0.43652671  5.01155335]]
----------------------------------------------------------
Accepted job 1
CommBox: Sent job 1
WARNING: cis already read from TC input file, skipping cis: yes
WARNING: cisnumstates already read from TC input file, skipping cisnumstates: 2
WARNING: cistarget already read from TC input file, skipping cistarget: 1
Finished job 1
Accepted job 2
CommBox: Sent job 2
WARNING: cis already read from TC input file, skipping cis: yes
WARNING: cisnumstates already read from TC input file, skipping cisnumstates: 2
Finished job 2
Accepted job 3
CommBox: Sent job 3

==================== PROPAGATION STEP 1 ====================
Time: 0.1000 fs
Electronic x: [1.35160498 0.08533418]
Electronic p: [1.26611707 1.15946133]
Adiabatic Pops: [1. 0.]
QM Atom 0 Coords (Bohr): [-0.81447191  6.54223141 -4.27078076]
MM Atom 0 Coords (Bohr): [-6.338141   -0.33637123  5.29501225]
--- DEBUG (models/qm.py): Data sent to TeraChem at step 1 ---
QM Coords (first 2 atoms, Bohr):
[[-0.81447319  6.54223371 -4.27078227]
 [ 0.23431687  7.31700713 -5.85435165]]
MM Coords (first 2 atoms, Bohr):
[[-6.33814203 -0.33637135  5.29501158]
 [-4.43516875 -0.43652771  5.01156292]]
----------------------------------------------------------
Finished job 3
Accepted job 4
CommBox: Sent job 4
....
