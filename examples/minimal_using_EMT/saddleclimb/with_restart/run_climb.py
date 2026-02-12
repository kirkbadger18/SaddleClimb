from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from ase.calculators.emt import EMT

calc = EMT()
init=read('../../init/opt.traj')
final=read('../../final/opt.traj')

climber = SaddleClimb(init, final, calc)
climber.climb(maxsteps=4)
