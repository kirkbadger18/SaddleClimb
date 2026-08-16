from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from mace.calculators import mace_mp


init=read('../../init/opt.traj')
final=read('../../final/opt.traj')
calc = mace_mp(model="small", default_dtype="float64")

climber = SaddleClimb(init, final, calc)
climber.climb()
