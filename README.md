# SaddleClimb:
## A path constrained minimum mode following saddle point search algorithm

This method combines the advantages of both single and double ended search methods for finding first order saddle points that connect
reactive intermedieates to one another. The atoms corresponding to an initial state are slowly stepped uphill along the minimum mode which has a positive
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
geometry files for the initial and final states of your reaction. Then you can use Saddleclimb to search for the first order saddle point connecting the two. As an example I have optimised a hydrogen adatom on Pt(111) in both the atop and fcc site, then used saddleclimb to find the saddlepoint connecting the two. Below is an example of how saddleclimb was used.

```python
from ase import Atoms, Atom
from ase.io import read
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
```

To see more details refer to this example and others in the [example](https://github.com/kirkbadger18/SaddleClimb/tree/local/example) folder
