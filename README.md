# SaddleClimb:
## A path biased minimum mode following saddle point search algorithm

This method combines the advantages of both single and double ended search methods for finding first order saddle points that connect
reactive intermediates to one another. At the start of the optimization, the atoms corresponding to an initial state are slowly stepped uphill in the direction of the average path. The average path is the difference between the final state and the initial state. At each step, an approximate hessian is updated using the TS-BFGS method. Once one of the eigenvectors switches to being negative, the rest of the optimization is done via. minimum mode following. This seems to work over a wide range of surface reactions from dissociation, abstraction, vdW dissociation. It seems to take somewhere between 60-200 gradient calls from an electronic structure calulator to achieve convergence of 0.01 eV/Angstrom. In comparison a NEB with 7 intermediate images might need 7*300 = 2100 gradient calls. Please test this out and let me know if it also works for you all.

## Installation
First clone this repository:

`git clone git@github.com:kirkbadger18/SaddleClimb.git`

Then activate your virtual environment of choice containing ASE, enter the repository directory, and build the package with:

`pip install ./`

## How to Use
This tool is meant to be used in combination with the Atomic Simulation Environment (ASE). It will take in ASE Atoms objects and ASE calculator objects. To instantiate a SaddleClimb object you will need an initial state (optimized), and a final state (also optimized), and a calculator. The initial and final states are ASE Atoms objects, and the calculator is an ASE calculator object. The following section of code is an example of how this would be set up for the diffusion of a carbon adatom from one site to another using the EMT calculator.

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
To see more details of this example see: [here](https://github.com/kirkbadger18/SaddleClimb/tree/main/examples/minimal_using_EMT)
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
To see more details of restarting a job see: [here](https://github.com/kirkbadger18/SaddleClimb/tree/main/examples/minimal_using_EMT/saddleclimb/with_restart)
## SaddleClimb output
The output from Saddleclimb is two files: climb.log, and climb.traj. In climb.log, the iteration number, energy, and fmax values are stored after each gradient call to the ASE calculator supplied. For the above example this looks like:
```
Iteration           Energy (eV)         Fmax (eV/A)         
0                   7.162017            0.000888            
1                   7.162922            0.030578            
2                   7.16546             0.05738             
3                   7.175275            0.157905            
4                   7.205316            0.469773            
5                   7.23593             0.751324            
6                   7.239443            0.729584            
7                   7.232197            0.624944            
8                   7.216657            0.178321            
9                   7.21549             0.0771              
10                  7.214951            0.08292             
11                  7.213808            0.047538            
12                  7.213432            0.037712            
13                  7.213204            0.043334            
14                  7.212673            0.049304            
15                  7.211994            0.037834            
16                  7.211815            0.027823            
17                  7.211764            0.026378            
18                  7.211704            0.018304            
19                  7.211657            0.009406
```
And climb.traj can be opened with `ase gui` and shows the geometry and energy of each step. The energy profiles typically look like:

<img width="1.5*294" height="1.5*191" alt="image" src="https://github.com/user-attachments/assets/57b35f13-9522-4236-b008-6b5060f946ed" />

## Contribution
If you would like to help fix bugs, add example data, or suggest features, or optional arguments, please make an issue first, then we can make pull requests to adress issues. This way no pull request comes out of nowhere suggesting changes that we may not be interrested in making.
