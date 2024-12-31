# Import the PyRosetta and FastRelax modules
import sys
from pyrosetta import *
from pyrosetta.rosetta.protocols.relax import FastRelax


# Initialize PyRosetta and load your protein structure
init()
pose = pose_from_pdb(sys.argv[1])

# set the score function
scorefxn = get_fa_scorefxn()

# Create a FastRelax object and use it to refine the structure
fast_relax = FastRelax()
fast_relax.set_scorefxn(scorefxn)
fast_relax.apply(pose)
pose.dump_pdb("./result_refine.pdb")