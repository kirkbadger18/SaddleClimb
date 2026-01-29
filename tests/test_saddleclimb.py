import numpy as np
import numpy.linalg as LA
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from saddleclimb import SaddleClimb


def generate_saddleclimb_object():
    calc = EMT()
    init = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
    final = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
    add_adsorbate(init, 'H', 1.5, 'fcc')
    add_adsorbate(final, 'H', 1.5, 'hcp')
    idx = list(range(18, 37))
    climber = SaddleClimb(init, final, calc, idx)
    return climber


def test___init__():
    climber = generate_saddleclimb_object()
    assert climber.atoms_initial
    assert type(climber.atoms_initial) is Atoms

    assert climber.atoms_final
    assert type(climber.atoms_final) is Atoms
    assert climber.indices
    assert type(climber.indices) is list
    assert climber.hessian is not None
    assert type(climber.hessian) is np.ndarray

    assert climber.calculator
    assert isinstance(climber.calculator, Calculator)
    assert climber.fmax
    assert type(climber.fmax) is float
    assert climber.maxstepsize
    assert type(climber.maxstepsize) is float
    assert climber.delta
    assert type(climber.delta) is float
    assert climber.logfile
    assert type(climber.logfile) is str
    assert climber.trajfile
    assert type(climber.trajfile) is str

    assert np.shape(climber.atoms_final) == np.shape(climber.atoms_initial)
    assert len(climber.indices) <= len(climber.atoms_initial)
    assert climber.hessian.shape[0] == climber.hessian.shape[1]
    assert climber.hessian.shape[0] == 3 * len(climber.indices)


def test_normalize():
    climber = generate_saddleclimb_object()
    vec = np.random.rand(5)
    normalized_vec = climber.normalize(vec)
    assert LA.norm(normalized_vec) == pytest.approx(1)
