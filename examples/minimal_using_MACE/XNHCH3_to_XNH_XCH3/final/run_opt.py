#!/usr/bin/env python3
from ase.io import read
from ase.optimize import BFGS
from mace.calculators import mace_mp

A = read('init.traj')
A.set_pbc(True)

A.calc = mace_mp(model="small", default_dtype="float64")
dyn = BFGS(A, trajectory='opt.traj', logfile='opt.log')
dyn.run(fmax=0.01)
