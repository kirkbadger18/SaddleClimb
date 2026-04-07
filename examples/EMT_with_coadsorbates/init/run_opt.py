#!/bin/sh
from ase import Atoms, Atom
from ase.io.trajectory import Trajectory
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.io import read,write

C=read('init.traj')
C.set_pbc(True)

C.calc = EMT()
dyn = BFGS(C, trajectory='opt.traj')
dyn.run(fmax=0.001)
