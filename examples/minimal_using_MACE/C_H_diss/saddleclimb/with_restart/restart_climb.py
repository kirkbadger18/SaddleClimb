from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from mace.calculators import mace_mp

calc = mace_mp(model="small", default_dtype="float64")
init=read('../../init/opt.traj')
final=read('../../final/opt.traj')
restarttraj = read('climb.traj')
climber = SaddleClimb(init, final, calc)
climber.restart_climb(restarttraj)
