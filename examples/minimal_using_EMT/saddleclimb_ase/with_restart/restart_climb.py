from ase.io import read
from ase.calculators.emt import EMT
from saddleclimb_ase import SaddleClimb

atoms_initial = read('../../init/opt.traj')
atoms_initial.calc = EMT()
atoms_final = read('../../final/opt.traj')

climber = SaddleClimb(
    atoms_initial, atoms_final,
    restart='climb.traj',
    append_trajectory=True,
)
climber.run(fmax=0.01)
