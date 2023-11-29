from ase.build import molecule
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import numpy as np
from ase.vibrations import Vibrations, VibrationsData
from ase.io import read
from ase.build import fcc111, add_adsorbate
from numpy.linalg import eigh
from ase.constraints import FixAtoms
from saddleclimb import SaddleClimb
from ase.calculators.vasp import Vasp

calculator=Vasp(xc='beef-vdw',
	encut=340.14, #25Ry
	luse_vdw=True,
	zab_vdw=-1.8867,
	kpts=(3,3,1),
	ismear=1,
	sigma=0.1,
	ibrion=-1,
	ispin=2,
	algo='Fast',
	lreal='Auto',
	ediff=1e-5,
	isym=0,) 

testmoli=read('testi2.traj')
testmolf=read('testf2.traj')

vib = Vibrations(testmoli, indices=[36])
vib.get_frequencies()
dat = vib.get_vibrations(testmoli)
H = dat.get_hessian_2d()

climber = SaddleClimb(testmoli, testmolf, [36], calculator, H)
climber.run()

