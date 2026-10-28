# SaddleClimb:
## A path-biased saddle-point search algorithm

This method combines the advantages of both single- and double-ended search methods for finding first-order saddle points that connect
reactive intermediates to one another. At the start of the optimization, the atoms corresponding to the reactant undergo a small step (at most `maxstep`) towards the middle of the estimated reaction path in Cartesian space. An approximate Hessian, which is initialized as a scaled identity matrix (`hessian_scale`), is updated using the TS-BFGS method (Bofill method). Next, one of two step types is taken: either a directed step is taken, or a free-climbing step is taken. A free-climbing step is one that takes a partitioned-rational function optimization step (P-RFO) which uses the lowest eigenvector of the approximate Hessian as the climbing direction. A directed-climb step first requires constructing a new basis with the desired climbing direction as the leftmost entry. The approximate Hessian is mapped into this basis, and the coupling terms between the climbing direction and descent direction are nulled out. Then the modified Hessian is mapped back to Cartesian coordinates, and a P-RFO step is taken that climbs along the desired direction and descends in every direction orthogonal to it. Both components of the P-RFO step share a single scaling, chosen so that no atom moves further than `maxstep`.

Directed steps are taken early on in the climb, and the direction in which to climb is determined by an estimate of the path. A path is drawn from the initial state, through the current structure, to the final state. The two supported path estimation types are linear and quadratic synchronous transit (`interp='qst'`, the default). The linear path is the straight line from the initial to the final state, moved to pass through the current structure. The QST path follows Halgren and Lipscomb (Chem. Phys. Lett. 49 (1977) 225): each interatomic distance is interpolated quadratically through its value in the initial, current, and final structures, and the positions at any point along the path are found by a weighted least-squares fit to those distances. For periodic systems the distances include periodic images within `r_cut`, weighted as in Govind et al. (Comput. Mater. Sci. 28 (2003) 250). The energy along the path is then estimated with a quintic that has minima at both end states and matches the current energy and its slope along the path. The climbing direction points from the current structure to the path structure at the maximum of this fit, or to the middle of the path if the fit has no interior maximum.

The method switches to free-climbing steps once at least `min_directed_steps` steps have been taken and the lowest eigenvalue of the unmodified Hessian has been negative for at least `persistence` consecutive steps, with its eigenvector following on from one step to the next. If the negative mode becomes positive again, or the lowest mode is replaced by a different one, the counter resets and the method switches back to directed steps. As a guard, if the climbing direction (oriented uphill) points away from both the initial and the final state, the climbing component of the step is dropped and only the descent component is taken. The climb finishes once the largest atomic force is below `fmax` and the structure has moved at least `min_travel` from the initial state, so that the initial state itself is never returned as the saddle.

This seems to work over a wide range of surface reactions such as dissociation, abstraction, vdW dissociation. In the benchmark of 45 surface reactions in [benchmarking](benchmarking) (full data on [Zenodo](https://zenodo.org/records/23740279)), 87 of 90 runs (each reaction run in both directions) found the correct saddle, and those took somewhere between 20-81 gradient calls (about 50 on average) from an electronic structure calculator to achieve convergence of 0.01 eV/Å. In comparison, on the 23 reactions where an NEB with 9 intermediate images found the same saddle, the NEB needed about 1100 gradient calls on average, roughly 23 times more. Please test this out and let me know if it also works for you all.

## Installation
If you intend to use this method, but not work on it, you can simply pip install the latest stable version. First activate your virtual environment of choice (venv, conda ...), then pip install:

`pip install git+https://github.com/kirkbadger18/SaddleClimb.git@v0.5.0`

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
init=read('../../init/opt.traj')
final=read('../../final/opt.traj')

climber = SaddleClimb(init, final, calc)
climber.climb()
```

Only atoms that move between the initial and final states are optimized; atoms that sit in the same place in both states are held fixed. The initial and final states are used in the frame they are given: each atom's position in the final state should be where that atom ends up from the initial state, so if an atom crosses a periodic boundary, place it on the same side of the cell in both states rather than wrapping it back. The climb can be stopped after a set number of steps with `climber.climb(maxsteps=N)`.

If the job gets canceled or fails for some reason, the calculation can be restarted using the `restart_climb()` method. The approximate Hessian and the rest of the climb state are stored with every image in climb.traj, so the restart continues from the last image exactly where it left off. This would look like:
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

If there are multiple coadsorbates that are perhaps moving throughout the reaction, SaddleClimb will include their movement in the path. You can ask SaddleClimb to target specific reactive atom indices using the `target_indices` argument. This is a list of atom indices which SaddleClimb should be using to assess the initial to final state direction. The other moving atoms are left out of the climbing direction and the path coordinate, but they still relax in the descent part of each step. The code for this would look like:
```python
from ase import Atoms, Atom
from ase.io import read
from saddleclimb import SaddleClimb
from ase.calculators.emt import EMT

calc = EMT()
init=read('../../init/opt.traj')
final=read('../../final/opt.traj')

climber = SaddleClimb(init, final, calc, target_indices=[36])
climber.climb()
```
To see an example of this, look [here](https://github.com/kirkbadger18/SaddleClimb/blob/main/examples/EMT_with_coadsorbates). In this example, we want to find the first-order saddle point for the diffusion of C on a Pt(111) surface, but there are other co-adsorbed carbon atoms, and one of the spectator carbon atoms is also moving from the initial to the final state. The same reaction is run with and without `target_indices` in the `with_target` and `without_target` folders.

There are also examples using a fine-tuned MACE potential [here](https://github.com/kirkbadger18/SaddleClimb/tree/main/examples/minimal_using_MACE) (C-H dissociation, C diffusion, and NHCH3 dissociation run in both directions). These need `mace-torch` installed.

## Optional arguments
The defaults below are the ones used in the benchmark. In most cases only `fmax`, `target_indices`, and the file names need to be changed.

| Argument | Default | Description |
| --- | --- | --- |
| `fmax` | `0.01` | Convergence criterion on the largest atomic force (eV/Å). |
| `target_indices` | `None` | Atom indices that define the climbing direction and path coordinate. By default all moving atoms are used. |
| `logfile` | `'climb.log'` | Name of the log file. |
| `trajfile` | `'climb.traj'` | Name of the trajectory file. |
| `interp` | `'qst'` | Path estimate used for directed steps: `'qst'` or `'linear'`. |
| `hessian_scale` | `50` | Diagonal of the initial approximate Hessian (eV/Å²). |
| `min_directed_steps` | `8` | Minimum number of directed steps before free climbing is allowed. |
| `persistence` | `3` | Number of consecutive steps the lowest Hessian mode must be negative before switching to free climbing. |
| `maxstep` | `0.2` | Largest distance any atom may move in one step (Å). |
| `r_cut` | `6` | Cutoff for the periodic image pairs used in the QST path (Å). Ignored for non-periodic systems. |
| `min_travel` | `0.1` | Minimum distance (largest per-atom displacement, Å) from the initial state before the climb may converge. |
| `a_max` | `1` | Largest P-RFO scaling factor. |
| `max_scaling_halvings` | `60` | Number of halvings tried when searching for a P-RFO scaling that fits within `maxstep`. |
| `delta_f` | `0.01` | Finite-difference spacing along the path used for the QST tangent. |

## SaddleClimb output
The output from Saddleclimb is two files: climb.log, and climb.traj. In climb.log, the iteration number, energy, and fmax values are stored after each gradient call to the ASE calculator supplied. For the above example this looks like:
```
Iteration           Energy (eV)         Fmax (eV/A)
0                   7.162017            0.000889
1                   7.174764            0.194455
2                   7.192255            0.118772
3                   7.206885            0.086192
4                   7.212939            0.079641
5                   7.212626            0.039489
6                   7.212334            0.029455
7                   7.212176            0.027813
8                   7.21205             0.02475
9                   7.211884            0.024118
10                  7.211774            0.019494
11                  7.211703            0.01678
12                  7.211676            0.00802
```
One extra calculation is made on the final state at the start of the climb to get its energy, which is used in the path energy estimate; it is not written to the log. And climb.traj can be opened with `ase gui` and shows the geometry and energy of each step. The energy profiles typically look like:

<img width="450" height="295" alt="climb_traj" src="https://github.com/user-attachments/assets/194f27ef-ac08-4bb5-8c0e-4e42095a6385" />

## Contribution
If you would like to help fix bugs, add example data, or suggest features, or optional arguments, please make an issue first, then we can make pull requests to adress issues. This way no pull request comes out of nowhere suggesting changes that we may not be interrested in making.
