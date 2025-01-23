from ase import Atoms, Atom
from ase.io import read
from numpy.linalg import eigh
from ase.constraints import FixAtoms
from saddleclimb import SaddleClimb
from ase.calculators.vasp import Vasp
from saddleclimb import SaddleClimb

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

init=read('../../../minimizations/atop/opt.traj')
final=read('../../../minimizations/hcp/opt.traj')

idx = list(range(18,37))
climber = SaddleClimb(init, final, calc, idx)
climber.test_run()
