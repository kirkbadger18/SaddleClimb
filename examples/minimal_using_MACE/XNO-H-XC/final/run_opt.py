#!/usr/bin/env python3
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_mp

C = read('init.traj')
C.set_pbc(True)

C.calc = mace_mp(model="small", default_dtype="float64")
dyn = BFGS(C, trajectory='opt.traj')
dyn.run(fmax=0.01)
