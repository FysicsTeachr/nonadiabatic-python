import sys
import parmed as pmd  # Import the ParmEd library

# Load TCPB wrapper
try:
    import pytcpb as tc
except:
    print("ERROR: Failed to import pytcpb in test_api.py")
    sys.exit(1)

# --- NEW: File loading and data extraction ---

# 1. Define input filenames
prmtop_file = "system.prmtop"
rst7_file = "system.rst7"
qmregion_file = "system.qmregion"

# 2. Read the QM atom indices from the .qmregion file
#    Subtract 1 because Amber indices are 1-based, but Python lists are 0-based.
with open(qmregion_file, 'r') as f:
    qm_indices = {int(line.strip()) for line in f if line.strip()}
print(f"Read {len(qm_indices)} QM atom indices from {qmregion_file}")

# 3. Load the Amber files using ParmEd
print(f"Loading structure from {prmtop_file} and coordinates from {rst7_file}")
structure = pmd.load_file(prmtop_file, rst7_file)

# 4. Initialize lists for QM and MM atoms
qmattypes = []
qmcoords = []
mmcoords = []
mmcharges = []

# 5. Separate atoms into QM and MM lists
for i, atom in enumerate(structure.atoms):
    # Flatten coordinates into a simple list [x1, y1, z1, x2, y2, z2, ...]
    coords = [structure.coordinates[i][0], structure.coordinates[i][1], structure.coordinates[i][2]]
    
    if i in qm_indices:
        # This is a QM atom
        qmattypes.append(atom.element_name) # Get element symbol (e.g., 'O', 'H')
        qmcoords.extend(coords)
    else:
        # This is an MM atom
        mmcharges.append(atom.charge)
        mmcoords.extend(coords)

print(f"Separated system into {len(qmattypes)} QM atoms and {len(mmcharges)} MM atoms.")
# --- End of new section ---


# Conversion parameter: Bohr to Angstrom
BohrToAng = 0.52917724924

# Set information about the server
host = "localhost"
port = 12345

# Other input variables
tcfile = "terachem.inp" # Make sure this file is updated!
globaltreatment = 0

# Attempts to connect to the TeraChem server
print(f" Attempting to connect to TeraChem server using host {host} and port {port}.")
status = tc.connect(host, port)
if (status == 0):
    print(" Successfully connected to TeraChem server.")
elif (status == 1):
    print(" ERROR: Connection to TeraChem server failed!")
    sys.exit(1)
else:
    print(" ERROR: Status on tc.connect function is not recognized!")
    sys.exit(1)

# Setup TeraChem
# The qmattypes are now read from the file
status =  tc.setup(tcfile,qmattypes)
if (status == 0):
    print(" TeraChem setup completed with success.")
else:
    print(f" ERROR: Failed to setup TeraChem (status {status})!")
    sys.exit(1)


# Convert all coordinates from Angstroms (from rst7) to Bohrs for TeraChem
for i in range(len(qmcoords)):
    qmcoords[i] /= BohrToAng
for i in range(len(mmcoords)):
    mmcoords[i] /= BohrToAng

# Compute energy and gradient
print("\nComputing energy and gradient...")
totenergy, qmgrad, mmgrad, status = tc.compute_energy_gradient(qmattypes,qmcoords,mmcoords,mmcharges,globaltreatment)
if (status == 0):
    print(" Computed energy and gradient with success.")
else:
    print(f" ERROR: Problem to compute energy and gradient (status {status})!")
    sys.exit(1)

# Print results
print("\n--- Results ---")
print(f"E = {totenergy:16.10f} Hartrees")
for i in range(len(qmattypes)):
    print(f"QM Grad({i+1:3d},:) = {qmgrad[3*i]:16.10f}{qmgrad[3*i+1]:16.10f}{qmgrad[3*i+2]:16.10f} Hartree/Bohr")
for i in range(int(len(mmcoords)/3)):
    print(f"MM Grad({i+1:3d},:) = {mmgrad[3*i]:16.10f}{mmgrad[3*i+1]:16.10f}{mmgrad[3*i+2]:16.10f} Hartree/Bohr")

# Finalizes variables on the TeraChem side
tc.finalize()
