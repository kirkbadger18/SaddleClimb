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


def generate_saddleclimb_object(interp='linear'):
    calc = EMT()
    init = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
    final = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
    add_adsorbate(init, 'H', 1.5, 'fcc')
    add_adsorbate(final, 'H', 1.5, 'hcp')
    idx = list(range(18, 37))
    climber = SaddleClimb(init, final, calc,
                          target_indices=idx, interp=interp)
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
    assert climber.maxstep
    assert type(climber.maxstep) is float
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


def test_pfro_uses_one_scaling_for_climb_and_descent():
    """One scaling is solved on the whole step and both parts use it.

    A soft climb mode with a large gradient along it forces the scaling
    below a_max, and the descent, which would fit unscaled, is scaled
    down with it.  The step is the sum of both parts at that one scaling
    and reaches maxstep.
    """
    climber = generate_saddleclimb_object()
    rng = np.random.default_rng(3)
    n = 12
    Q, _ = LA.qr(rng.standard_normal((n, n)))
    eigs = np.concatenate(([-0.5], rng.uniform(5, 30, n - 1)))
    B_opt = Q @ np.diag(eigs) @ Q.T
    vmax, vmin = Q[:, 0], Q[:, 1:]
    g = 3.0 * vmax + 0.01 * vmin @ rng.standard_normal(n - 1)
    climber._climb_mode = vmax

    def climb(a):
        return climber._get_scaled_climb_step(B_opt, g, vmax, a)

    def descend(a):
        return climber._get_scaled_descend_step(B_opt, g, vmin, a)

    assert climber._get_maxstep(climb(climber.a_max)) > climber.maxstep
    assert climber._get_maxstep(descend(climber.a_max)) < climber.maxstep
    a = climber._get_pfro_scaling(lambda s: climb(s) + descend(s),
                                  climber.maxstep)
    assert a < climber.a_max

    step = climber._get_pfro_step(B_opt, g)
    assert_allclose(climber._get_maxstep(step), climber.maxstep, rtol=1e-2)
    assert climber._get_maxstep(step) <= climber.maxstep + 1e-9
    assert_allclose(step, climb(a) + descend(a), rtol=1e-2, atol=1e-3)
    across = step - np.dot(step, vmax) * vmax
    assert climber._get_maxstep(across) < climber._get_maxstep(
        descend(climber.a_max))


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
        climber._climb_mode = vmax

        climber._climbing = True
        climbed = climber._get_pfro_step(B_opt, g)
        assert abs(np.dot(climbed, vmax)) > 1e-8

        climber._climbing = False
        step = climber._get_pfro_step(B_opt, g)
        assert_allclose(np.dot(step, vmax), 0, atol=1e-12)
        assert np.dot(g, step) < 0
        assert (climber._get_maxstep(step)
                <= climber.maxstep + 1e-9)


def test_climb_guard_reads_the_gradient():
    """The guard stops only when the gradient points away from both ends."""
    climber = generate_saddleclimb_object()
    idx = climber.indices
    climber._pos_i_1D = climber.atoms_initial.positions[idx, :].reshape(-1)
    climber._pos_f_1D = climber.atoms_final.positions[idx, :].reshape(-1)
    chord = climber._pos_f_1D - climber._pos_i_1D
    dhat = climber.normalize(chord)
    pos_1D = climber._pos_i_1D + 0.5 * chord
    dxi = climber._pos_i_1D - pos_1D
    dxf = climber._pos_f_1D - pos_1D

    # Uphill toward either end: still climbing.
    assert climber._is_climbing(dhat, dxi, dxf)
    assert climber._is_climbing(-dhat, dxi, dxf)

    # Uphill across the chord, away from both ends: stop.
    basis, _ = LA.qr(dhat.reshape(-1, 1), mode='complete')
    off = basis[:, 1]
    far = climber._pos_i_1D + 0.5 * chord + 0.5 * off
    dxi = climber._pos_i_1D - far
    dxf = climber._pos_f_1D - far
    assert not climber._is_climbing(off, dxi, dxf)


def test_tripped_guard_nulls_the_free_climb_step_and_bias_is_unguarded():
    """A free step failing the guard keeps the lowest mode but nulls the
    climb component; a directed step is not guarded at all.

    Past the final endpoint and off the chord along +w, a gradient along
    +w leads away from both endpoints.
    """
    climber = generate_saddleclimb_object()
    idx = climber.indices
    n = 3 * len(idx)
    climber._pos_i_1D = climber.atoms_initial.positions[idx, :].reshape(-1)
    climber._pos_f_1D = climber.atoms_final.positions[idx, :].reshape(-1)
    chord = climber._pos_f_1D - climber._pos_i_1D
    dhat = climber.normalize(chord)
    basis, _ = LA.qr(dhat.reshape(n, 1), mode='complete')
    w = basis[:, 1]
    far = climber._pos_i_1D + 1.3 * chord + 0.5 * w
    g = 0.5 * dhat + 0.3 * w
    assert not climber._is_climbing(g, climber._pos_i_1D - far,
                                    climber._pos_f_1D - far)

    for directed in (True, False):
        eigs = np.full(n, 10.0)
        eigs[0], eigs[1] = (5.0 if directed else -5.0), 1.0
        B = basis @ np.diag(eigs) @ basis.T
        if not directed:
            climber._step_count = climber.min_directed_steps
            climber._prev_mode = LA.eigh(B)[1][:, 0]
            climber._prev_eig = -5.0
        B_opt = climber._get_B_opt(B, g, far)
        step = climber._get_step(B_opt, g)
        v = climber._climb_mode
        assert_allclose(abs(np.dot(v, dhat)), 1, atol=1e-10)
        assert bool(climber._free) == (not directed)
        assert bool(climber._climbing) == directed
        if directed:
            assert abs(np.dot(step, v)) > 1e-6
        else:
            assert_allclose(np.dot(step, v), 0, atol=1e-12)


def test_directed_b_opt_keeps_signs_and_prfo_climbs_convex_bias():
    """B_opt only decouples the bias; P-RFO still climbs it.

    In a convex basin the bias keeps its positive curvature u^T B u and
    every other eigenvalue keeps its sign.  P-RFO then steps uphill
    along the bias and downhill across it.
    """
    climber = generate_saddleclimb_object()
    idx = climber.indices
    n = 3 * len(idx)
    climber._pos_i_1D = climber.atoms_initial.positions[idx, :].reshape(-1)
    climber._pos_f_1D = climber.atoms_final.positions[idx, :].reshape(-1)
    dhat = climber.normalize(climber._pos_f_1D - climber._pos_i_1D)
    rng = np.random.default_rng(4)
    A = rng.standard_normal((n, n))
    B = A @ A.T + 2 * np.eye(n)
    g = 0.05 * dhat + 0.3 * climber.normalize(rng.standard_normal(n))
    pos_1D = climber._pos_i_1D + 0.2 * (climber._pos_f_1D
                                        - climber._pos_i_1D)

    B_opt = climber._get_B_opt(B, g, pos_1D)
    assert_allclose(abs(np.dot(climber._climb_mode, dhat)), 1, atol=1e-12)
    assert np.all(LA.eigvalsh(B_opt) > 0)
    assert_allclose(B_opt @ dhat, (dhat @ B @ dhat) * dhat, atol=1e-10)

    step = climber._get_step(B_opt, g)
    v = climber._climb_mode
    assert np.dot(g, v) * np.dot(step, v) > 0
    across = step - np.dot(step, v) * v
    assert np.dot(g, across) < 0


def test_target_indices_set_the_qst_bias():
    """Only target atoms enter the path coordinate and the QST bias.

    A surface Pt atom moves between the end states but is not a
    target, so it may not shift p, and the bias holds it still.
    """
    init = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
    final = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
    add_adsorbate(init, 'H', 1.5, 'fcc')
    add_adsorbate(final, 'H', 1.5, 'hcp')
    final.positions[27] += [0.1, 0.05, 0.1]
    climber = SaddleClimb(init, final, EMT(), target_indices=[36],
                          interp='qst')
    assert climber.indices == [27, 36]
    assert climber._bias_atoms == [36]
    idx = climber.indices
    climber._pos_i_1D = init.positions[idx].reshape(-1)
    climber._pos_f_1D = final.positions[idx].reshape(-1)

    mid = init.positions.copy()
    mid[36] = 0.7 * init.positions[36] + 0.3 * final.positions[36]
    p = climber._get_path_coordinate(mid)
    assert_allclose(p, 0.3, atol=1e-12)
    shifted = mid.copy()
    shifted[27] = final.positions[27]
    assert_allclose(climber._get_path_coordinate(shifted), p, atol=1e-12)

    bias = climber._get_bias_vector(mid[idx].reshape(-1))
    assert_allclose(bias[:3], 0, atol=1e-12)
    assert LA.norm(bias[3:]) > 0

    with pytest.raises(ValueError):
        SaddleClimb(init, final, EMT(), target_indices=[0])


def test_tripped_guard_is_a_one_off():
    """A tripped guard nulls only the step it trips on.

    The next step, where the guard passes, climbs again at once, and the
    step stays free throughout.
    """
    climber = generate_saddleclimb_object()
    climber._step_count = climber.min_directed_steps
    idx = climber.indices
    n = 3 * len(idx)
    climber._pos_i_1D = climber.atoms_initial.positions[idx, :].reshape(-1)
    climber._pos_f_1D = climber.atoms_final.positions[idx, :].reshape(-1)
    chord = climber._pos_f_1D - climber._pos_i_1D
    dhat = climber.normalize(chord)
    basis, _ = LA.qr(dhat.reshape(n, 1), mode='complete')
    w = basis[:, 1]
    eigs = np.full(n, 10.0)
    eigs[0] = -5.0
    B = basis @ np.diag(eigs) @ basis.T
    climber._prev_mode = LA.eigh(B)[1][:, 0]
    climber._prev_eig = -5.0

    # Past the final endpoint, a gradient along +w leads away from both.
    far = climber._pos_i_1D + 1.3 * chord + 0.5 * w
    mid = climber._pos_i_1D + 0.5 * chord
    climbing = []
    for pos_1D, g in [(mid, 0.1 * dhat + 0.3 * w), (far, 0.5 * dhat + 0.3 * w),
                      (mid, 0.1 * dhat + 0.3 * w),
                      (mid, 0.1 * dhat + 0.3 * w)]:
        climber._get_B_opt(B, g, pos_1D)
        assert climber._free
        climbing.append(climber._climbing)
    assert climbing == [True, False, True, True]


def test_free_needs_negative_lowest_mode_that_matches_the_last_one():
    """Free only if the lowest mode is negative and best-matched.

    Before ``min_directed_steps`` every step is directed.  The first
    step after has no previous B, so it is directed too.  A repeat of
    the same B is free.  A negative lowest mode that is not the previous
    lowest mode, or a lowest mode that is not negative, is directed.
    """
    climber = generate_saddleclimb_object()
    idx = climber.indices
    n = 3 * len(idx)
    climber._pos_i_1D = climber.atoms_initial.positions[idx, :].reshape(-1)
    climber._pos_f_1D = climber.atoms_final.positions[idx, :].reshape(-1)
    chord = climber._pos_f_1D - climber._pos_i_1D
    dhat = climber.normalize(chord)
    basis, _ = LA.qr(dhat.reshape(n, 1), mode='complete')
    w = basis[:, 1]
    pos_1D = climber._pos_i_1D + 0.5 * chord
    g = 0.1 * dhat + 0.3 * w

    eigs = np.full(n, 10.0)
    eigs[0] = -5.0
    B = basis @ np.diag(eigs) @ basis.T
    # Too early: directed whatever B looks like.
    climber._get_B_opt(B, g, pos_1D)
    climber._step_count = climber.min_directed_steps - 1
    climber._get_B_opt(B, g, pos_1D)
    assert not climber._free
    climber._step_count = climber.min_directed_steps
    climber._prev_mode = None
    climber._get_B_opt(B, g, pos_1D)
    assert not climber._free
    climber._get_B_opt(B, g, pos_1D)
    assert climber._free

    # The negative mode turns into a different direction: no match.
    eigs_2 = np.full(n, 10.0)
    eigs_2[1] = -5.0
    B_2 = basis @ np.diag(eigs_2) @ basis.T
    climber._get_B_opt(B_2, g, pos_1D)
    assert not climber._free

    # Same mode, but no longer negative.
    eigs_3 = np.full(n, 10.0)
    eigs_3[1] = 1.0
    B_3 = basis @ np.diag(eigs_3) @ basis.T
    climber._get_B_opt(B_3, g, pos_1D)
    assert not climber._free

    # Previous mode matches but its eigenvalue was not negative.
    climber._prev_mode = basis[:, 0]
    climber._prev_eig = 0.5
    climber._get_B_opt(B, g, pos_1D)
    assert not climber._free
