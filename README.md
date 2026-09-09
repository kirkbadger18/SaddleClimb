# SaddleClimb:
## A path-biased minimum-mode-following saddle-point search algorithm

This method combines the advantages of both single- and double-ended search methods for finding first-order saddle points that connect
reactive intermediates to one another. At the start of the optimization, the atoms corresponding to an initial state are slowly stepped uphill in the direction of the average path. The average path is the difference between the final state and the initial state. At each step, an approximate Hessian is updated using the TS-BFGS method. Once one of the eigenvectors switches to being negative (and stays negative for at least 3 steps), the rest of the optimization is done via partitioned rational-function optimization. This seems to work over a wide range of surface reactions from dissociation, abstraction, vdW dissociation. It seems to take somewhere between 20-100 gradient calls from an electronic structure calculator to achieve convergence of 0.01 eV/Å. In comparison a NEB with 7 intermediate images might need 7×300 = 2100 gradient calls. Please test this out and let me know if it also works for you all.

## Installation
If you intend to use this method, but not work on it, you can simply pip install the latest stable version. First activate your virtual environment of choice (venv, conda ...), then pip install:

`pip install git+https://github.com/kirkbadger18/SaddleClimb.git@v0.2.0`

If you want to install the developer version, first clone this repository:

`git clone git@github.com:kirkbadger18/SaddleClimb.git`

Then activate your virtual environment of choice, enter the directory (`cd SaddleClimb`), and install Saddleclimb as editable:

`pip install -e ./`

Now you are set to simply import this package from any directory with this conda environment active. Follow the steps below on how to use this package.

## How to Use
This tool is meant to be used in combination with the Atomic Simulation Environment (ASE). It will take in ASE Atoms objects and ASE calculator objects. To instantiate a SaddleClimb object you will need an initial state (optimized), and a final state (also optimized), and a calculator. The initial and final states are ASE Atoms objects, and the calculator is an ASE calculator object. The following section of code is a snippet from an example on how this would be set up for the diffusion of a carbon adatom from one site to another using the EMT calculator. To see more details of this example see [here](https://github.com/kirkbadger18/SaddleClimb/tree/main/examples/minimal_using_EMT).

```python
from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from ase.calculators.emt import EMT

calc = EMT()
init=read('../init/opt.traj')
final=read('../final/opt.traj')

climber = SaddleClimb(init, final, calc)
climber.climb()
```

If the job gets canceled or fails for some reason, the calculation can be restarted using the `restart_climb()` method. This would look like:
```python
from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from ase.calculators.emt import EMT

calc = EMT()
init=read('../../init/opt.traj')
final=read('../../final/opt.traj')
restarttraj = read('climb.traj')
climber = SaddleClimb(init, final, calc)
climber.restart_climb(restarttraj)
```
To see more details of restarting a job see [here](https://github.com/kirkbadger18/SaddleClimb/tree/main/examples/minimal_using_EMT/saddleclimb/with_restart)

If there are multiple coadsorbates that are perhapse moving throughout the reaction, SaddleClimb will include their movement in the average path. You can ask SaddleClimb to target specific reactive atom indices using the `target_indices` argument. This is a list of atom indices for which SaddleClimb should be using to assess the initial to final state direction. The code for this would look like:
```python
from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from ase.calculators.emt import EMT

calc = EMT()
init=read('../init/opt.traj')
final=read('../final/opt.traj')

climber = SaddleClimb(init, final, calc, target_indices=[36])
climber.climb()
```
To see an example of this, look [here](https://github.com/kirkbadger18/SaddleClimb/blob/main/examples/EMT_with_coadsorbates). In this example, we want to find the first-order saddle point for the diffusion of C on a Pt(111) surface, but there are other co-adsorbed carbon atoms, and one of the spectator carbon atoms is also moving from the initial to the final state.

## SaddleClimb output
The output from Saddleclimb is two files: climb.log, and climb.traj. In climb.log, the iteration number, energy, and fmax values are stored after each gradient call to the ASE calculator supplied. For the above example this looks like:
```
Iteration           Energy (eV)         Fmax (eV/A)
0                   7.162017            0.000888
1                   7.162922            0.037283
2                   7.165438            0.078442
3                   7.174929            0.192475
4                   7.204055            0.480515
5                   7.229134            0.663377
6                   7.23271             0.622422
7                   7.225272            0.477201
8                   7.215206            0.085159
9                   7.214696            0.073657
10                  7.213615            0.059447
11                  7.213202            0.040847
12                  7.212853            0.050085
13                  7.21229             0.047477
14                  7.211839            0.026553
15                  7.211731            0.022867
16                  7.211694            0.018647
17                  7.211659            0.010225
18                  7.211644            0.004605
```
And climb.traj can be opened with `ase gui` and shows the geometry and energy of each step. The energy profiles typically look like:

<img width="450" height="295" alt="climb_traj" src="https://github.com/user-attachments/assets/194f27ef-ac08-4bb5-8c0e-4e42095a6385" />

## Contribution
If you would like to help fix bugs, add example data, or suggest features, or optional arguments, please make an issue first, then we can make pull requests to adress issues. This way no pull request comes out of nowhere suggesting changes that we may not be interrested in making.
