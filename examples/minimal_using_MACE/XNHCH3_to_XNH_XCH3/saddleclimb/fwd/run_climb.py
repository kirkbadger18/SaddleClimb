from ase.io import read
from saddleclimb import SaddleClimb
from mace.calculators import MACECalculator

init = read('../../init/opt.traj')
final = read('../../final/opt.traj')
calc = MACECalculator(model_paths='../../../mace_finetuned.model',
                      default_dtype='float64')

climber = SaddleClimb(init, final, calc)
climber.climb(maxsteps=300)
