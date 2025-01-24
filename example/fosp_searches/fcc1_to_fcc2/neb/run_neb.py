#!/bin/sh
from ase import Atoms, Atom
from ase.io.trajectory import Trajectory
from ase.calculators.vasp import Vasp
from ase.optimize import BFGS
from ase.io import read,write
from ase.mep import interpolate,NEB

init=read('../../../minimizations/fcc_1/opt.traj')
final=read('../../../minimizations/fcc_2/opt.traj')
images = [init]
#images=read('neb.traj@-9:')

for i in range(7):
    image=init.copy()
    calculator = Vasp(directory='./image_{:d}'.format(int(i+1)),
                      xc='beef-vdw',
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
	                  isym=0)
    image.calc = calculator
#    image.calc=calculator
    images.append(image)
images.append(final)
neb=NEB(images)
neb.interpolate()
dyn = BFGS(neb, trajectory='neb.traj', logfile='neb.log')
dyn.run(fmax=0.01)
