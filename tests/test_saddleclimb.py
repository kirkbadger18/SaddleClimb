import numpy as np
import numpy.linalg as LA
from numpy.testing import assert_allclose
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from saddleclimb import SaddleClimb
from pathlib import Path
import tempfile
import copy


def generate_saddleclimb_object():
    calc = EMT()
    init = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
    final = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
    add_adsorbate(init, 'H', 1.5, 'fcc')
    add_adsorbate(final, 'H', 1.5, 'hcp')
    idx = list(range(18, 37))
    climber = SaddleClimb(init, final, calc, idx)
    return climber


def test__init__():
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


def test_initialize_logging():
    climber = generate_saddleclimb_object()
    n_str = 'Iteration'.ljust(20)
    E_str = 'Energy (eV)'.ljust(20)
    F_str = 'Fmax (eV/A)'.ljust(20)
    log_string = n_str + E_str + F_str + "\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.txt"
        climber.logfile = file_path
        climber._initialize_logging()
        with open(file_path, 'r') as log:
            lines = log.readlines()
        assert file_path.exists()
        assert file_path.is_file()
        assert lines[0] == log_string
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.txt"
        climber.logfile = file_path
        with open(file_path, 'w') as log:
            log.write('tmpstring')
        climber._initialize_logging()
        assert file_path.exists()
        assert file_path.is_file()
        with open(file_path, 'r') as log:
            lines = log.readlines()
        assert 'tmpstring' not in lines[0]
        assert lines[0] == log_string


def test_log():
    climber = generate_saddleclimb_object()
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.txt"
        climber.logfile = file_path
        with open(file_path, 'w') as log:
            log.write('firstline\n')
        climber._log('secondline\n')
        with open(file_path, 'r') as log:
            lines = log.readlines()
        assert file_path.exists()
        assert file_path.is_file()
        assert lines[0] == 'firstline\n'
        assert lines[1] == 'secondline\n'


def test_get_log_string():
    climber = generate_saddleclimb_object()
    E, n, Fmax = 1, 1, 1
    n_str = str(n).ljust(20)
    E_str = str(np.round(E, 6)).ljust(20)
    F_str = str(np.round(Fmax, 6)).ljust(20)
    log_string = n_str + E_str + F_str
    test_log_string = climber._get_log_string(n, E, Fmax)
    print(type(test_log_string))
    assert isinstance(test_log_string, str)
    assert test_log_string == log_string


def test_get_F():
    climber = generate_saddleclimb_object()
    atoms = climber.atoms_initial.copy()
    atoms.calc = climber.calculator
    test_atoms = copy.deepcopy(atoms)
    F = climber._get_F(test_atoms)
    assert isinstance(F, np.ndarray)
    assert F.shape == atoms.positions.shape
    assert_allclose(test_atoms.positions, atoms.positions)


def test_initialize_atoms():
    climber = generate_saddleclimb_object()
    atoms = climber.atoms_initial.copy()
    atoms_test, idx_test, B_test = climber._initialize_atoms()
    assert isinstance(atoms_test, Atoms)
    assert isinstance(idx_test, list)
    assert isinstance(B_test, np.ndarray)
    assert atoms == atoms_test
    assert climber.calculator.results == atoms_test.calc.results
    assert climber.indices == idx_test
    assert_allclose(climber.hessian, B_test)
