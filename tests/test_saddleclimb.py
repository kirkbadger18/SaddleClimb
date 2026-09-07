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


def test_pfro_step_nulls_ascent_when_not_climbing():
    """The guard drops the vmax component; the rest still relaxes."""
    climber = generate_saddleclimb_object()
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = 12
        Q, _ = LA.qr(rng.standard_normal((n, n)))
        eigs = rng.uniform(0.2, 30, n)
        eigs[0] = -eigs[0]
        B_opt = Q @ np.diag(np.sort(eigs)) @ Q.T
        g = rng.standard_normal(n) * 0.3
        _, vecs = LA.eigh(B_opt)
        vmax = vecs[:, 0]

        climber._climbing = True
        climbed = climber._get_pfro_step(B_opt, g)
        assert abs(np.dot(climbed, vmax)) > 1e-8

        climber._climbing = False
        step = climber._get_pfro_step(B_opt, g)
        assert_allclose(np.dot(step, vmax), 0, atol=1e-12)
        assert np.dot(g, step) < 0
        assert climber._get_maxstep(step) <= climber.maxstepsize + 1e-9


def test_climb_guard_reads_ascent_direction_not_gradient():
    """The guard tests the direction the ascent component moves.

    The step along vmax carries the sign of g.vmax, so the ascent
    direction is +/- vmax, not g.  Off the endpoint chord both endpoint
    dots pick up a shared perpendicular term, so a gradient dominated by
    that term -- the near-convergence case -- reads as "away from both"
    no matter what the climb is doing.  The ascent direction does not.
    """
    climber = generate_saddleclimb_object()
    idx = climber.indices
    n = 3 * len(idx)
    climber._pos_i_1D = climber.atoms_initial.positions[idx, :].reshape(-1)
    climber._pos_f_1D = climber.atoms_final.positions[idx, :].reshape(-1)
    climber._directed = False

    half = (climber._pos_f_1D - climber._pos_i_1D) / 2
    chord = climber.normalize(half)
    basis, _ = LA.qr(chord.reshape(n, 1), mode='complete')
    off = basis[:, 1]
    # Sit off the chord, so both endpoints lie back along -off.
    pos_1D = climber._pos_i_1D + half + 0.5 * off
    dxi = climber._pos_i_1D - pos_1D
    dxf = climber._pos_f_1D - pos_1D

    def hessian_with_lowest_mode(vec):
        vecs, _ = LA.qr(vec.reshape(n, 1), mode='complete')
        return vecs @ np.diag(np.concatenate(([-5.0],
                                              np.full(n - 1, 5.0)))) @ vecs.T

    # Gradient dominated by +off: both endpoint dots go negative, so the
    # gradient rule stops.  The ascent direction lies along the chord and
    # still points at an endpoint, so the climb continues.
    g = 0.01 * chord + 8.0 * off
    assert np.dot(g, dxi) < 0 and np.dot(g, dxf) < 0
    climber._get_B_opt(hessian_with_lowest_mode(chord), g, pos_1D, 50)
    assert climber._climbing

    # Converse: the gradient still points at the final endpoint, but the
    # ascent direction is +off, which leads away from both.
    g = 8.0 * chord + 0.01 * off
    assert np.dot(g, dxf) > 0
    climber._get_B_opt(hessian_with_lowest_mode(off), g, pos_1D, 50)
    assert not climber._climbing


def test_climb_guard_is_live_during_directed_climb():
    """The guard applies to the biased mode as well as the free one.

    While ``_directed`` holds, the QR surgery makes dhat an exact
    eigenvector, so the ascent direction is +/- dhat.  With
    t = dhat.(pos - pos_i) and L = dhat.(pos_f - pos_i), the endpoint
    dots are -t and L - t, so the guard fires exactly on overshoot --
    past the final endpoint when climbing along +dhat, behind the
    initial one when climbing along -dhat.  Between the endpoints the
    chord always points at one of them and the climb must continue.
    """
    climber = generate_saddleclimb_object()
    idx = climber.indices
    n = 3 * len(idx)
    climber._pos_i_1D = climber.atoms_initial.positions[idx, :].reshape(-1)
    climber._pos_f_1D = climber.atoms_final.positions[idx, :].reshape(-1)
    chord = climber._pos_f_1D - climber._pos_i_1D
    dhat = climber.normalize(chord)

    # dhat distinctly lowest and B positive definite, so _get_B_opt takes
    # the directed branch and climbs dhat itself.
    basis, _ = LA.qr(dhat.reshape(n, 1), mode='complete')
    eigs = np.concatenate(([1.0], np.full(n - 1, 5.0)))
    B = basis @ np.diag(eigs) @ basis.T

    def guard(frac, sign):
        pos_1D = climber._pos_i_1D + frac * chord
        g = sign * 0.5 * dhat + 0.02 * basis[:, 1]
        climber._directed = True
        B_opt = climber._get_B_opt(B, g, pos_1D, 50)
        vmax = LA.eigh(B_opt)[1][:, 0]
        assert_allclose(abs(np.dot(vmax, dhat)), 1.0, atol=1e-10)
        return climber._climbing

    # Between the endpoints the guard cannot fire, either orientation.
    for frac in [0.25, 0.5, 0.9]:
        assert guard(frac, +1)
        assert guard(frac, -1)

    # Past the final endpoint, climbing along +dhat leads away from both.
    assert not guard(1.05, +1)
    assert not guard(1.3, +1)
    # Behind the initial endpoint, the same is true of -dhat.
    assert not guard(-0.1, -1)
    # The opposite orientation still points back at an endpoint.
    assert guard(1.3, -1)
    assert guard(-0.1, +1)
