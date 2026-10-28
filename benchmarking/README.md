# SaddleClimb benchmark (v0.5.0)

This folder holds a benchmark of SaddleClimb v0.5.0 on 45 surface reactions computed with DFT, and a comparison against NEB on the reactions where NEB found the same saddle point. The question it answers is how many gradient calls SaddleClimb needs to find the saddle point that connects a known reactant and product, compared to a nudged elastic band.

## Contents

| File | Description |
| --- | --- |
| `Total_results.csv` | Gradient calls for SaddleClimb (forward and reverse) and NEB on all 45 reactions, with NEB notes, reaction family, and whether each run completed (converged to the correct saddle). |
| `NEB_comparison.csv` | The 23 reactions with a converged NEB that found the same saddle, with averages and the SaddleClimb speedup. |

The full calculation data (ASE run scripts, Quantum ESPRESSO inputs and outputs, trajectories, and logs for every optimization, Sella, IRC, SaddleClimb, and NEB run) is too large for this repository and is archived on Zenodo: https://zenodo.org/records/23740279. It extracts to `Benchmarking/<metal>/<reaction>/`, one folder per reaction, where `X` in a folder name marks a surface site.

## Systems

- **Pt(111)**: 40 reactions of C, N, O, and H adsorbates (C-H, N-H, O-H, C-C, C-N, C-O, N-N, and N-O bond breaking, plus abstraction reactions), on a four-layer 3x3 slab.
- **Co(0001)**: 5 abstraction reactions of the form \*C + \*R <-> \*CR' + \*R'', run spin polarized.

## Calculation settings

All calculations use Quantum ESPRESSO 7.1 through ASE, with the BEEF-vdW functional, PAW pseudopotentials from pslibrary 1.0.0, a 40 Ry wavefunction cutoff (410 Ry density cutoff), Marzari-Vanderbilt smearing of 0.02 Ry, a 3x3x1 k-point grid, and an SCF threshold of 1e-8 Ry. The Co calculations are spin polarized (`nspin=2`). Every saddle search and the NEB are converged to `fmax = 0.01` eV/Å.

## Workflow

For reactions 1-35 and the Co reactions, each reaction was run in this order:

1. Geometry optimizations of initial guesses for the reactant and product.
2. A reference saddle point found with Sella (`eta=0.01`, `fmax=0.001`), started from a guessed saddle structure.
3. Sella IRCs (`dx=0.1`, `fmax=0.01`) from the reference saddle in both directions. The last IRC images are the reactant and product used by everything below, so every method starts from minima that are known to connect through the reference saddle.
4. SaddleClimb forward, from the reactant to the product, with all defaults, `SaddleClimb(init, final, calc)`.
5. SaddleClimb reverse: the same climb with the end states swapped, which tests whether the result depends on which end the climb starts from.
6. Where an NEB was run: NEB with 9 intermediate images (improved tangent, `k=5`, BFGS), followed by a climbing-image NEB restarted from the last NEB band.

A SaddleClimb run counts as a success when it converges to the same saddle as Sella, connecting the same reactant and product.

Reactions 36-40 (listed under "Starting from new guess" in `Total_results.csv`) show SaddleClimb used the way it would be in practice, without a known saddle to start from. The reactant and product are the optimized guesses, SaddleClimb is run between them, and IRCs are started from the SaddleClimb saddle to confirm that it connects the intended reactant and product. These reactions have no Sella run.

## Results

The gradient-call columns count calls to the DFT calculator. For SaddleClimb they are the number of steps in `climb.log`. For NEB they are summed over all moving images of the NEB and the CI-NEB. A `+` after an NEB count means the NEB did not reach the right saddle within those calls; the reason is in the NEB notes column.

The "completed" columns of `Total_results.csv` mark whether each run converged to the correct saddle. SaddleClimb completed 87 of the 90 runs. The three failures are all on Co(0001): reaction 41 reverse, and reaction 42 in both directions (125 and 169 gradient calls). The 87 runs that completed took between 20 and 81 gradient calls, 51 on average.

An NEB counts as completed when it found the same saddle as SaddleClimb. Reaction 3 is counted as completed: its NEB and SaddleClimb saddles differ by about 50 meV, more than for the other reactions, probably because the two were converged to slightly different forces.

On the 23 reactions in `NEB_comparison.csv`, SaddleClimb averaged 49 gradient calls forward and 47 reverse, against 1099 for NEB, a speedup of about 23 times. Five more reactions have an NEB but are left out of the average. In reactions 2, 10, 11, and 15 the NEB did not find the right saddle: the band was held up by diffusion or rotation modes of the fragments. In reaction 39 most of the band was held up by CH4 rotation. Reactions 27 to 38 and the Co reactions have no NEB.
