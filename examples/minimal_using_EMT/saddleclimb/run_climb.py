from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from ase.calculators.emt import EMT

calc = EMT()
init=read('../init/opt.traj')
final=read('../final/opt.traj')
idx = list(range(18,37))

climber = SaddleClimb(init, final, calc, idx)
climber.run()
