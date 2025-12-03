#!/bin/sh
from ase import Atoms, Atom
from ase.io.trajectory import Trajectory
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.optimize import BFGS
from ase.io import read,write
from ase.mep import interpolate,NEB

#init=read('../../minimizations/init/opt.traj')
#final=read('../../minimizations/final/opt.traj')
#images = [init]
images=read('../neb.traj@-9:')



for i in range(7):
    #image=init.copy()
    pwx = '/oscar/runtime/software/external/quantum-espresso/7.1-git/bin/pw.x'
    profile = EspressoProfile(
        command='mpirun -n 48 --bind-to core {}'.format(pwx),
        pseudo_dir='/users/kbadger1/espresso/pseudo/pslibrary.1.0.0/pbe/PSEUDOPOTENTIALS/',
    )

    input_data = dict(
        input_dft='beef-vdw',
        occupations='smearing',
        smearing='marzari-vanderbilt',
        degauss=0.001,
        ecutwfc=40, #opt setting
        ecutrho = 410,
        nosym=True,
        nspin=1,
        mixing_mode='local-TF',
        tprnfor=True,
        conv_thr = 1e-8,
        )

    calc = Espresso(
                pseudopotentials=dict(
                                    Pt='Pt.pbe-n-kjpaw_psl.1.0.0.UPF',
                                    C='C.pbe-n-kjpaw_psl.1.0.0.UPF',
                                    O='O.pbe-nl-kjpaw_psl.1.0.0.UPF',
                                    H='H.pbe-kjpaw_psl.1.0.0.UPF',
                                    N='N.pbe-n-kjpaw_psl.1.0.0.UPF',),
                input_data=input_data,
                profile=profile,
                kpts=(5, 5, 1), #opt setting
                directory='./image_{:d}'.format(int(i+1)),
                )

    images[i+1].calc = calc
    #images.append(image)
    
#images.append(final)
neb=NEB(images, climb=True)
#neb.interpolate()
dyn = BFGS(neb, trajectory='neb.traj', logfile='neb.log')
dyn.run(fmax=0.01)
