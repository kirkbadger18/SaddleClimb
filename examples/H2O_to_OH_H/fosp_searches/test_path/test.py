from ase.io import read
from saddleclimb import *

images = read('neb.traj@-9:')
idx = list(range(18,39))

for i in range(7):
    img = images[i+1]
    F = img.calc.results['forces']
    F_norm = normalize(F[idx, :].reshape(-1))
    init = images[0]
    final = images[-1]
    pos_1D = img.positions[idx, :].reshape(-1)
    climber = SaddleClimb(init, final, None, idx)
    path = climber.get_ellipse_tangent(pos_1D)
    dot = np.dot(path,-F_norm)
    print(dot)
