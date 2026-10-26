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


def seed_streak(climber, B):
    """Put the climber one step short of a free climb on B's negative mode."""
    climber._step_count = climber.min_directed_steps
    climber._prev_mode = LA.eigh(B)[1][:, 0]
    climber._streak = climber.persistence - 1


def unit(v):
    return v / LA.norm(v)


def hop_climber():
    """H hopping 1.2 -> 3.6 A in x past a Pt in a 5 A cell: it ends
    nearer the Pt's periodic image at x = 5 than the Pt itself."""
    init = Atoms('PtH', positions=[[0, 0, 0], [1.2, 0, 1.0]],
                 cell=[5, 5, 20], pbc=[True, True, False])
    final = init.copy()
    final.positions[1] = [3.6, 0, 1.0]
    return SaddleClimb(init, final, EMT())


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
    assert climber.logfile
    assert type(climber.logfile) is str
    assert climber.trajfile
    assert type(climber.trajfile) is str

    assert np.shape(climber.atoms_final) == np.shape(climber.atoms_initial)
    assert len(climber.indices) <= len(climber.atoms_initial)
    assert climber.hessian.shape[0] == climber.hessian.shape[1]
    assert climber.hessian.shape[0] == 3 * len(climber.indices)


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
    """
    A soft, steep climb mode forces the one scaling below a_max; the
    descent, which would fit unscaled, shrinks with it, and the step is
    the sum of both parts at that scaling and reaches maxstep.
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


def test_climb_guard_reads_the_uphill_climb_mode():
    """The guard stops only when the uphill climb direction points away
    from both ends; the sign of the mode itself does not matter."""
    climber = generate_saddleclimb_object()
    chord = climber._pos_f_1D - climber._pos_i_1D
    dhat = unit(chord)
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


def test_tripped_guard_nulls_the_climb_step_free_or_directed():
    """A step failing the guard, free or directed, keeps its climb mode
    but nulls the climb component.

    Past the final endpoint and off the chord along +w, a gradient along
    +w leads away from both endpoints.
    """
    climber = generate_saddleclimb_object()
    idx = climber.indices
    n = 3 * len(idx)
    chord = climber._pos_f_1D - climber._pos_i_1D
    dhat = unit(chord)
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
        step = climber._get_pfro_step(B_opt, g)
        v = climber._climb_mode
        assert_allclose(abs(np.dot(v, dhat)), 1, atol=1e-10)
        assert bool(climber._free) == (not directed)
        assert not climber._climbing
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
    dhat = unit(climber._pos_f_1D - climber._pos_i_1D)
    rng = np.random.default_rng(4)
    A = rng.standard_normal((n, n))
    B = A @ A.T + 2 * np.eye(n)
    g = 0.05 * dhat + 0.3 * unit(rng.standard_normal(n))
    pos_1D = climber._pos_i_1D + 0.2 * (climber._pos_f_1D
                                        - climber._pos_i_1D)

    B_opt = climber._get_B_opt(B, g, pos_1D)
    assert_allclose(abs(np.dot(climber._climb_mode, dhat)), 1, atol=1e-12)
    assert np.all(LA.eigvalsh(B_opt) > 0)
    assert_allclose(B_opt @ dhat, (dhat @ B @ dhat) * dhat, atol=1e-10)

    step = climber._get_pfro_step(B_opt, g)
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

    mid = init.positions.copy()
    mid[36] = 0.7 * init.positions[36] + 0.3 * final.positions[36]
    p = climber._get_path_coordinate(mid)
    assert_allclose(p, 0.3, atol=1e-12)
    shifted = mid.copy()
    shifted[27] = final.positions[27]
    assert_allclose(climber._get_path_coordinate(shifted), p, atol=1e-12)

    _, t, _ = climber._get_path_frame(mid[idx].reshape(-1))
    assert_allclose(t[:3], 0, atol=1e-12)
    assert LA.norm(t[3:]) > 0

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
    chord = climber._pos_f_1D - climber._pos_i_1D
    dhat = unit(chord)
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
    """
    Free only after ``persistence`` steps with a following negative lowest
    mode.  Further negative modes are ignored, a different lowest mode or
    none restart the count, and ``min_directed_steps`` holds it directed.
    """
    climber = generate_saddleclimb_object()
    climber.min_directed_steps = 0
    climber.persistence = 3
    idx = climber.indices
    n = 3 * len(idx)
    chord = climber._pos_f_1D - climber._pos_i_1D
    dhat = unit(chord)
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
    rng = np.random.default_rng(7)
    A = rng.standard_normal((n, n))
    B = A @ A.T + 2 * np.eye(n)
    off = 0.02 * rng.standard_normal(n)
    pos = (climber._pos_i_1D
           + scale * (climber._pos_f_1D - climber._pos_i_1D) + off)
    g = 0.3 * rng.standard_normal(n)
    return climber, B, g, pos


@pytest.mark.parametrize('interp', ['linear', 'qst'])
def test_path_climb_search_fills_the_trust_radius(interp):
    """With a large gradient the step fills the trust radius and climbs."""
    climber, B, g, pos = directed_setup(interp)
    g = 8 * g
    B_opt = climber._get_B_opt(B, g, pos)
    step = climber._get_pfro_step(B_opt, g)
    assert_allclose(climber._get_maxstep(step), climber.maxstep, rtol=2e-2)
    assert abs(step @ climber._climb_mode) > 0


def test_fit_without_interior_maximum_climbs_to_the_middle():
    """
    Energy and slope on the smoothstep E_P (10f^3 - 15f^4 + 6f^5), which
    rises to P with no interior maximum, give no fitted f*; the climb
    heads for the path structure at f* = 0.5.
    """
    climber, B, _, pos = directed_setup('qst')
    f, t, path_at = climber._get_path_frame(pos)
    E_P = 1.0
    climber._E_initial, climber._E_final = 0.0, E_P
    climber._E_cur = E_P * (10*f**3 - 15*f**4 + 6*f**5)
    G_f = E_P * (30*f**2 - 60*f**3 + 30*f**4)
    assert climber._get_path_energy_max(f, G_f) is None
    climber._get_B_opt(B, G_f * t / (t @ t), pos)
    chord = path_at(0.5) - pos
    assert_allclose(abs(climber._climb_mode @ chord) / LA.norm(chord),
                    1, atol=1e-10)


@pytest.mark.parametrize('interp', ['linear', 'qst'])
def test_directed_climb_mode_is_the_chord_to_f_star_and_an_eigenvector(interp):
    """
    Without energies f* falls back to 0.5; the climb mode is the unit
    chord to the path there, and an exact eigenvector of B_opt with its
    own curvature.
    """
    climber, B, g, pos = directed_setup(interp)
    B_opt = climber._get_B_opt(B, g, pos)
    _, _, path_at = climber._get_path_frame(pos)
    chord = path_at(0.5) - pos
    v = climber._climb_mode
    assert_allclose(abs(v @ chord) / LA.norm(chord), 1, atol=1e-10)
    assert_allclose(B_opt @ v, (v @ B @ v) * v, atol=1e-10)


@pytest.mark.parametrize('interp', ['linear', 'qst'])
def test_shrinking_alpha_keeps_the_climb_direction(interp):
    """A smaller trust radius shortens the climb along the same line."""
    climber, B, g, pos = directed_setup(interp)
    g = 8 * g
    B_opt = climber._get_B_opt(B, g, pos)
    v = climber._climb_mode
    climbs = []
    for maxstep in (0.4, 0.1, 0.03):
        climber.maxstep = maxstep
        step = climber._get_pfro_step(B_opt, g)
        assert climber._get_maxstep(step) <= maxstep + 1e-9
        climbs.append(np.dot(step, v) * v)
    for c in climbs[1:]:
        assert_allclose(abs(c @ climbs[0]) / (LA.norm(c) * LA.norm(climbs[0])),
                        1, atol=1e-9)
    assert LA.norm(climbs[2]) < LA.norm(climbs[1]) < LA.norm(climbs[0])


def test_final_structure_is_taken_in_the_frame_given():
    """An atom placed one lattice vector further on in the final structure
    is taken to travel that far: it is not moved to a nearer image."""
    init = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
    add_adsorbate(init, 'H', 1.5, 'fcc')
    final = init.copy()
    h = len(init) - 1
    travel = np.array([0.3, 0.2, 0.0]) + init.cell[1]
    final.positions[h] += travel
    climber = SaddleClimb(init, final, EMT(), interp='linear')

    assert climber.indices == [h]
    assert_allclose(climber.atoms_final.positions, final.positions)
    dx, _ = climber._get_initial_step()
    assert_allclose(dx[0] @ travel, LA.norm(dx[0]) * LA.norm(travel),
                    rtol=1e-6)


def test_image_pairs_keep_one_image_from_reactant_to_product():
    """A pair keeps the same image at both ends: an adsorbate that hops
    from one surface atom towards that atom's periodic image is
    interpolated away from the first and towards the second, as two
    image pairs, not as one pair whose length never changes."""
    climber = hop_climber()
    pairs = climber._get_image_pairs(climber.atoms_initial.positions)
    r_R = climber._image_distances(climber.atoms_initial.positions, pairs)
    r_P = climber._image_distances(climber.atoms_final.positions, pairs)
    ends = set(zip(np.round(r_R, 10), np.round(r_P, 10)))
    # Pt at the origin: bonded, then 3.6 A away.
    assert (round(np.hypot(1.2, 1), 10), round(np.hypot(3.6, 1), 10)) in ends
    # Pt's image at x = 5: 3.8 A away, then bonded.
    assert (round(np.hypot(3.8, 1), 10), round(np.hypot(1.4, 1), 10)) in ends


def test_qst_fit_gradient_matches_finite_differences(monkeypatch):
    """The analytic gradient of the image-pair fit matches S itself."""
    import sys
    climber = generate_saddleclimb_object('qst')
    pos = climber.atoms_initial.positions.copy()
    pos[-1] += [0.4, 0.3, 0.1]
    pairs = climber._get_image_pairs(pos)
    r = climber._image_distances(climber.atoms_final.positions, pairs)
    module = sys.modules[SaddleClimb.__module__]
    real, captured = module.minimize, {}

    def spy(fun, x0, **kw):
        captured['fun'], captured['x0'] = fun, x0
        return real(fun, x0, **kw)
    monkeypatch.setattr(module, 'minimize', spy)
    climber._fit_positions(pairs, r, climber._cut_weight(r), pos)
    fun, x = captured['fun'], captured['x0'] + 0.05
    _, grad = fun(x)
    h = 1e-6
    fd = np.array([(fun(x + h * e)[0] - fun(x - h * e)[0]) / (2 * h)
                   for e in np.eye(len(x))])
    assert_allclose(grad, fd, rtol=1e-5, atol=1e-8)


def test_molecule_keeps_every_pair_at_full_weight():
    """Without periodicity every pair enters once, uncut, as in eq. (2)."""
    init = Atoms('H3', positions=[[0, 0, 0], [0.8, 0, 0], [30, 0, 0]])
    final = init.copy()
    final.positions[2] = [29, 0, 0]
    climber = SaddleClimb(init, final, EMT())
    a, b, T = climber._get_image_pairs(init.positions)
    assert len(a) == 3 and not T.any()
    assert climber._cut_weight(np.array([30.0]))[0] == 1 / 30.0**4


def test_displacements_to_the_ends_are_plain_differences():
    """An atom more than half a cell from the product still measures
    towards the product the path runs to, not towards a nearer image of
    it, in both the path coordinate and the guard."""
    climber = hop_climber()
    idx = climber.indices
    pos = climber.atoms_initial.positions.copy()
    pos[1] = [-1.3, 0, 1.0]
    assert_allclose(climber._get_path_coordinate(pos), 2.5 / (2.5 + 4.9))

    # Uphill along -x points away from the reactant and from the product
    # the path runs to; the product's nearer image lies just to -x.
    B = 30 * np.eye(3 * len(idx))
    climber._get_B_opt(B, np.array([-1.0, 0, 0]), pos[idx].reshape(-1))
    assert not climber._climbing


def test_restart_structure_is_taken_as_written():
    """A restart continues from the positions in the trajectory, which the
    climb wrote in its own frame."""
    climber = generate_saddleclimb_object()
    restart = climber.atoms_initial.copy()
    restart.positions[-1] += [0.2, 0.1, 0] + restart.cell[0]
    restart.info['saddleclimb_hessian'] = climber.hessian.tolist()
    climber._restart_trajectory = restart
    atoms, _, _ = climber._initialize_atoms_restart()
    assert_allclose(atoms.positions, restart.positions)


@pytest.mark.parametrize('interp', ['linear', 'qst'])
def test_first_step_heads_for_the_middle_of_the_path(interp):
    """The first step is the chord to the path structure at f = 0.5,
    scaled so the furthest-moving atom moves ``maxstep``."""
    climber = generate_saddleclimb_object(interp)
    _, dx_1D = climber._get_initial_step()
    _, _, path_at = climber._get_path_frame(climber._pos_i_1D)
    chord = climber._hold_non_targets(path_at(0.5) - climber._pos_i_1D)
    assert_allclose(dx_1D @ chord, LA.norm(dx_1D) * LA.norm(chord),
                    rtol=1e-10)
    assert_allclose(climber._get_maxstep(dx_1D),
                    min(climber.maxstep, climber._get_maxstep(chord)),
                    rtol=1e-10)
