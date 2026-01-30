#!/bin/sh
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.constraints import FixAtoms
from ase.io import read,write
from ase.vibrations import Vibrations
import numpy as np

syst = read('../saddleclimb/climb.traj')

pwx = '/oscar/rt/9.6/25/spack/x86_64_v3/quantum-espresso-7.1-5jaxcfd5jl2ab6vodmr6jszzeh4tk6fz/bin/pw.x'
profile = EspressoProfile(
    command='mpirun -n 128 --bind-to core {}'.format(pwx),
    pseudo_dir='/users/kbadger1/espresso/pseudo/pslibrary.1.0.0/pbe/PSEUDOPOTENTIALS/',
)

input_data = dict(
        input_dft='beef-vdw',
        occupations='smearing',
        smearing='marzari-vanderbilt',
        degauss=0.02,
        ecutwfc=50, #opt setting
        ecutrho = 410,
        nosym=True,
        nspin=1,
        mixing_mode='local-TF',
        tprnfor=True,
        conv_thr=1e-8,
        calculation = "ensemble",
        )

calc = Espresso(
            pseudopotentials=dict(Pt='Pt.pbe-n-kjpaw_psl.1.0.0.UPF',
                                        C='C.pbe-n-kjpaw_psl.1.0.0.UPF',
                                        O='O.pbe-nl-kjpaw_psl.1.0.0.UPF',
                                        H='H.pbe-kjpaw_psl.1.0.0.UPF',
                                        N='N.pbe-n-kjpaw_psl.1.0.0.UPF',),
            input_data=input_data,
            profile=profile,
            kpts=(5, 5, 1), #opt setting
            )

no_slab = 36
total=syst.get_global_number_of_atoms()
adsorbate=list(np.arange(no_slab,total,1))
syst.calc = calc
vib = Vibrations(syst, indices=adsorbate)
vib.run()
vib.summary()
vib.write_mode()
