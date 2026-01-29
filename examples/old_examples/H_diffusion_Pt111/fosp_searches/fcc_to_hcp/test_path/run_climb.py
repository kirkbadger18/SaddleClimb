import numpy as np
from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from ase.calculators.espresso import Espresso, EspressoProfile
from saddleclimb import *

init=read('../../../minimizations/fcc/opt.traj')
final=read('../../../minimizations/hcp/opt.traj')
idx = list(range(18,37))
calc=None
climber = SaddleClimb(init, final, calc, idx)
images = read('neb.traj@-9::')


for i in range(len(images)-2):
    pos_1D = images[i+1].positions[idx,:].reshape(-1)
    path = climber.get_ellipse_tangent(pos_1D)
    F = normalize(images[i+1].calc.results['forces'][idx,:].reshape(-1))
    dot = np.dot(-F,path)
    print(dot)
#climber.run()
