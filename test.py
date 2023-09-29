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

#testmoli = read('POSCAR_10')
#del testmoli.constraints
slab = fcc111('Pt', size=(3,3,4))
add_adsorbate(slab, 'H', 1.5, 'ontop')
slab.center(vacuum=10.0, axis=2)
testmoli=slab.copy()
ci = FixAtoms(indices=[atom.index for atom in testmoli if atom.symbol == 'Pt'])
testmolf=testmoli.copy()
cf = FixAtoms(indices=[atom.index for atom in testmolf if atom.symbol == 'Pt'])
testmolf[36].position+=[1.4, 2.0, 0]
testmoli.set_constraint(ci)
testmolf.set_constraint(cf)
calc = EMT()
testmoli.calc = calc
dyn = BFGS(testmoli, trajectory='testi.traj')
dyn.run(fmax=0.05)
vib = Vibrations(testmoli, indices=[36], delta=1e-3, nfree=4)
vib.run()
vib.get_frequencies()
dat = vib.get_vibrations(testmoli)
H = dat.get_hessian_2d()


testmolf.calc = calc
dyn2 = BFGS(testmolf, trajectory='testf.traj')
dyn2.run(fmax=0.05)

climber = SaddleClimb(testmoli, testmolf, [36], calc, H)
climber.run()



