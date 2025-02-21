# SaddleClimb:
## A path constrained mode following saddle point search algorithm

This method combines the advantages of both single and double ended search methods for finding first order saddle points that connect
reactive intermediates to one another. The atoms corresponding to an initial state are slowly stepped uphill along the mode which has the largest
dot product with the reaction path. This path is not known exactly, but instead is fitted to the equation of an ellipse, where the verticies correspond to the initial and final states of the reaction. This method uses the final state coordinates to impose a path to guide the molecule uphill and towards the final state, but never constructs a string as other double ended methods to. This makes this method much faster, while retaining the robustness of a double
ended method.

## Installation
First clone this repository:

` git clone git@github.com:kirkbadger18/SaddleClimb.git`

Then add to your `~/.bashrc` file:

`export PATH=/path/to/saddleclimb/repo/saddleclimb.py:$PATH`

Lastly restart your shell, or source your `~/.bashrc`

## How to Use
This tool is meant to be used in combination with the Atomic Simulation Environment (ASE), so make sure that is installed. First you will need to have 
geometry files for the initial and final states of your reaction. Then you can use Saddleclimb to search for the first order saddle point connecting the two. As an example I have an initial state with *CH on Pt(111) and a final state where the *CH has dissociated to form *C and *H. I then used saddleclimb to find the first order saddle point connecting the two. Below is an example of how saddleclimb was used.

```python
from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from ase.calculators.espresso import Espresso, EspressoProfile
from saddleclimb_test import SaddleClimb
from ase.calculators.emt import EMT

pwx = '/oscar/runtime/software/external/quantum-espresso/7.1-git/bin/pw.x'
profile = EspressoProfile(
    command='mpirun -n 96 --bind-to core {}'.format(pwx),
    pseudo_dir='/users/kbadger1/espresso/pseudo/pslibrary.1.0.0/pbe/PSEUDOPOTENTIALS/',
)

input_data = dict(
        input_dft='beef-vdw',
        occupations='smearing',
        smearing='marzari-vanderbilt',
        degauss=0.02,
        ecutwfc=40, #opt setting
        ecutrho = 410,
        nosym=True,
        nspin=1,
        mixing_mode='local-TF',
        tprnfor=True,
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

                )

init=read('../../minimizations/init/opt.traj')
final=read('../../minimizations/final/opt.traj')
idx = list(range(18,38))

climber = SaddleClimb(init, final, calc, idx)
climber.run()
```

To see more details refer to this example and others in the [example](https://github.com/kirkbadger18/SaddleClimb/tree/main/examples/CH_to_C_H) folder
