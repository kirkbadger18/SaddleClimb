#!/bin/sh
from ase import Atoms, Atom
from ase.io.trajectory import Trajectory
from ase.calculators.vasp import Vasp
from ase.optimize import BFGS
from ase.constraints import FixAtoms, FixBondLength
from ase.io import read,write

C=read('init.traj')
C.set_pbc(True)

calc=Vasp(xc='beef-vdw',
    encut=680.29, 
	luse_vdw=True,
	zab_vdw=-1.8867,
	kpts=(5,5,1),
	ismear=1,
	sigma=0.1,
	ibrion=-1,
	ispin=1,
	algo='Fast',
	lreal='Auto',
	ediff=1e-5,
	isym=0,
    ) 

C.calc = calc
dyn = BFGS(C, trajectory='opt.traj')
dyn.run(fmax=0.01)
