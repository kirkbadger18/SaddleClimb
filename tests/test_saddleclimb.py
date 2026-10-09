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


def seed_streak(climber, B):
    """Put the climber one step short of a free climb on B's negative mode."""
    climber._step_count = climber.min_directed_steps
    climber._prev_mode = LA.eigh(B)[1][:, 0]
    climber._streak = climber.persistence - 1


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
    """The guard stops only when the uphill climb direction points away
    from both ends; the sign of the mode itself does not matter."""
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
    assert climber._is_climbing(dhat, dhat, dxi, dxf)
    assert climber._is_climbing(-dhat, dhat, dxi, dxf)
    assert climber._is_climbing(dhat, -dhat, dxi, dxf)

    # Uphill across the chord, away from both ends: stop.
    basis, _ = LA.qr(dhat.reshape(-1, 1), mode='complete')
    off = basis[:, 1]
    far = climber._pos_i_1D + 0.5 * chord + 0.5 * off
    dxi = climber._pos_i_1D - far
    dxf = climber._pos_f_1D - far
    assert not climber._is_climbing(off, off, dxi, dxf)
    assert not climber._is_climbing(-off, off, dxi, dxf)


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
    assert not climber._is_climbing(dhat, g, climber._pos_i_1D - far,
                                    climber._pos_f_1D - far)

    for directed in (True, False):
        eigs = np.full(n, 10.0)
        eigs[0], eigs[1] = (5.0 if directed else -5.0), 1.0
        B = basis @ np.diag(eigs) @ basis.T
        if not directed:
            seed_streak(climber, B)
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
    seed_streak(climber, B)

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


def test_free_needs_a_persistent_lowest_mode():
    """Free only after ``persistence`` steps with a persistent lowest mode.

    Each of the last ``persistence`` steps needs a negative lowest
    eigenvalue, and each one's lowest mode must overlap most with the
    next one's.  Further negative modes are ignored; a different lowest
    mode or none start the count over, and ``min_directed_steps`` holds
    the climb directed whatever the streak.
    """
    climber = generate_saddleclimb_object()
    climber.min_directed_steps = 0
    assert climber.persistence == 3
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

    def hessian(negative, value=-5.0):
        eigs = np.full(n, 10.0)
        eigs[negative] = value
        return basis @ np.diag(eigs) @ basis.T

    def free_after(B, step_count=100):
        climber._step_count = step_count
        climber._get_B_opt(B, g, pos_1D)
        return bool(climber._free), climber._streak

    B = hessian([0])
    assert free_after(B) == (False, 1)
    assert free_after(B) == (False, 2)
    assert free_after(B) == (True, 3)
    assert free_after(B) == (True, 4)

    # A second, shallower negative mode is ignored: the count goes on.
    B2 = hessian([0, 1])
    B2 = B2 + 4.0 * np.outer(basis[:, 1], basis[:, 1])
    assert free_after(B2) == (True, 5)

    # A different lowest mode follows on from nothing.
    assert free_after(hessian([1])) == (False, 1)
    assert free_after(hessian([1])) == (False, 2)
    assert free_after(hessian([1])) == (True, 3)

    # No negative mode at all.
    assert free_after(hessian([1], 1.0)) == (False, 0)

    # min_directed_steps holds the climb directed while the streak builds.
    climber.min_directed_steps = 5
    assert free_after(B, step_count=0) == (False, 1)
    assert free_after(B, step_count=0) == (False, 2)
    assert free_after(B, step_count=0) == (False, 3)
    assert free_after(B, step_count=climber.min_directed_steps) == (True, 4)

    with pytest.raises(ValueError):
        SaddleClimb(climber.atoms_initial, climber.atoms_final, EMT(),
                    persistence=0)


def directed_setup(interp, scale=0.2):
    """A climber part-way along its path and a convex B, so the step is
    directed (``_step_count`` is still below ``min_directed_steps``)."""
    climber = generate_saddleclimb_object(interp)
    idx = climber.indices
    n = 3 * len(idx)
    climber._pos_i_1D = climber.atoms_initial.positions[idx, :].reshape(-1)
    climber._pos_f_1D = climber.atoms_final.positions[idx, :].reshape(-1)
    rng = np.random.default_rng(7)
    A = rng.standard_normal((n, n))
    B = A @ A.T + 2 * np.eye(n)
    off = 0.02 * rng.standard_normal(n)
    pos = (climber._pos_i_1D
           + scale * (climber._pos_f_1D - climber._pos_i_1D) + off)
    g = 0.3 * rng.standard_normal(n)
    return climber, B, g, pos


def descent_across(climber, B_opt, g, a):
    basis, _ = LA.qr(climber._climb_mode.reshape(-1, 1), mode='complete')
    return climber._get_scaled_descend_step(B_opt, g, basis[:, 1:], a)


@pytest.mark.parametrize('interp', ['linear', 'qst'])
def test_path_climb_with_given_alpha_is_the_path_structure_at_f_star(interp):
    """The climb is the path structure at f* minus the current one.

    f* is the P-RFO maximum on the gradient and curvature projected
    onto the path tangent, and the rest of the step is the usual descent
    across the tangent, at the same alpha.
    """
    climber, B, g, pos = directed_setup(interp)
    B_opt = climber._get_B_opt(B, g, pos)
    assert not climber._free
    f, t, nodes, path_at = climber._path
    assert_allclose(abs(climber._climb_mode @ climber.normalize(t)), 1,
                    atol=1e-10)
    a = 0.15
    f_star = np.clip(f + climber._get_scaled_path_step(
        t @ B_opt @ t, t @ g, a), 0, 1)
    expected = path_at(f_star) - pos + descent_across(climber, B_opt, g, a)
    if climber._get_maxstep(expected) > climber.maxstep:
        expected *= climber.maxstep / climber._get_maxstep(expected)
    assert_allclose(climber._get_pfro_step(B_opt, g, a), expected,
                    atol=1e-10)


@pytest.mark.parametrize('interp', ['linear', 'qst'])
def test_path_climb_search_fills_the_trust_radius(interp):
    """The searched scaling uses the whole trust radius, and the step
    matches the path structure at the f* of that scaling.

    A large gradient makes the trust radius bind.  The polynomial that
    stands in for the path during the search is checked against solving
    the path at the f* it ends on.
    """
    climber, B, g, pos = directed_setup(interp)
    g = 8 * g
    B_opt = climber._get_B_opt(B, g, pos)
    f, t, nodes, path_at = climber._path
    step = climber._get_pfro_step(B_opt, g)
    assert_allclose(climber._get_maxstep(step), climber.maxstep, rtol=2e-2)

    # The climb part of the step, as a move along f, lands on the path.
    descent = LA.lstsq(
        LA.qr(climber._climb_mode.reshape(-1, 1), mode='complete')[0][:, 1:],
        step, rcond=None)[0]
    across = LA.qr(climber._climb_mode.reshape(-1, 1),
                   mode='complete')[0][:, 1:] @ descent
    climb = step - across
    assert LA.norm(climb) > 0
    f_new = f + (climb @ t) / (t @ t)
    assert_allclose(path_at(np.clip(f_new, 0, 1)) - pos, climb, atol=2e-3)


def test_path_climb_cannot_pass_the_end_structures():
    """A huge gradient along the path stops f* at the final structure."""
    climber, B, g, pos = directed_setup('linear', scale=0.5)
    climber.maxstep = 100.0
    f, t, nodes, path_at = climber._get_path_frame(pos)
    g = 1e4 * t / LA.norm(t)
    B_opt = climber._get_B_opt(B, g, pos)
    step = climber._get_pfro_step(B_opt, g, 1.0)
    # The linear path runs through the current structure parallel to the
    # chord, so f* = 1 is the rest of the chord from here.
    assert_allclose(step, (1 - f) * t, atol=1e-6)


def test_path_climb_off_gives_the_straight_climb():
    """With ``path_climb=False`` the climb is the old step along the bias."""
    climber, B, g, pos = directed_setup('qst')
    climber.path_climb = False
    B_opt = climber._get_B_opt(B, g, pos)
    a = 0.15
    v = climber._climb_mode
    expected = (climber._get_scaled_climb_step(B_opt, g, v, a)
                + descent_across(climber, B_opt, g, a))
    if climber._get_maxstep(expected) > climber.maxstep:
        expected *= climber.maxstep / climber._get_maxstep(expected)
    assert_allclose(climber._get_pfro_step(B_opt, g, a), expected,
                    atol=1e-12)
