from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from mace.calculators import MACECalculator

calc = MACECalculator(model_paths='../../../mace_finetuned.model',
                      default_dtype='float64')
init=read('../../init/opt.traj')
final=read('../../final/opt.traj')
restarttraj = read('climb.traj')
climber = SaddleClimb(init, final, calc, method='newton')
climber.restart_climb(restarttraj)
