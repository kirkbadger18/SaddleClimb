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


def test_pfro_scales_climb_and_descent_separately():
    """Each partition is scaled against maxstep on its own.

    A soft climb mode with a large gradient along it has to be scaled
    down, while a small descent fits unscaled at a_max.  The descent
    must therefore not be shortened by the climb's scaling, only by the
    final truncation of the sum, which scales both parts equally.
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
    a_climb = climber._get_pfro_scaling(climb, climber.maxstep)
    assert a_climb < climber.a_max
    assert_allclose(climber._get_maxstep(climb(a_climb)), climber.maxstep,
                    rtol=1e-2)

    step = climber._get_pfro_step(B_opt, g)
    assert climber._get_maxstep(step) <= climber.maxstep + 1e-9
    f = np.dot(step, vmax) / np.dot(climb(a_climb), vmax)
    assert 0 < f <= 1 + 1e-12
    across = step - np.dot(step, vmax) * vmax
    assert_allclose(across, f * descend(climber.a_max), atol=1e-12)


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
    assert climber._is_climbing(chord, g, dxi, dxf)

    # Converse: the gradient still points at the final endpoint, but the
    # ascent direction is +off, which leads away from both.  Asserted on
    # the guard itself.  Which mode gets selected is a separate question
    # from what the guard reads.
    g = 8.0 * chord + 0.01 * off
    assert np.dot(g, dxf) > 0
    assert not climber._is_climbing(off, g, dxi, dxf)


@pytest.mark.parametrize('directed', [True, False])
def test_guard_falls_back_to_bias_and_never_nulls(directed):
    """A free step failing the guard climbs the bias instead; bias steps
    are not guarded at all.

    B's lowest mode w is not the bias, and ascent along the bias and
    along w both lead away from both endpoints.  Free, w fails the guard
    and the step falls back to the bias; directed, the bias is climbed
    unguarded.  Either way the bias is the climb mode and the step still
    has a component along it.
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
    eigs[0], eigs[1] = 5.0, (1.0 if directed else -1.0)
    B = basis @ np.diag(eigs) @ basis.T
    climber._step_count = (0 if directed
                           else climber.min_directed_steps)

    # Past the final endpoint and off the chord along +w: ascent along
    # +dhat and along +w both lead away from both endpoints.
    pos_1D = climber._pos_i_1D + 1.3 * chord + 0.5 * w
    g = 0.5 * dhat + 0.3 * w
    dxi = climber._pos_i_1D - pos_1D
    dxf = climber._pos_f_1D - pos_1D
    assert not climber._is_climbing(w, g, dxi, dxf)
    assert not climber._is_climbing(dhat, g, dxi, dxf)

    B_opt = climber._get_B_opt(B, g, pos_1D)
    assert_allclose(abs(np.dot(climber._climb_mode, dhat)), 1, atol=1e-10)
    assert not climber._free
    step = climber._get_step(B_opt, g)
    assert abs(np.dot(step, dhat)) > 1e-6

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


def test_directed_for_min_steps_then_follows_lowest_sign():
    """The first min_directed_steps steps are directed whatever B says.

    After that a step is directed while B's lowest eigenvalue is
    non-negative and free while it is negative, switching back and
    forth with the sign.
    """
    climber = generate_saddleclimb_object()
    climber.min_directed_steps = 2
    directed = []
    for step, lowest in enumerate((-1, -1, -1, 0, 2, -1e-6, 1)):
        climber._step_count = step
        directed.append(climber._is_directed(lowest))
    assert directed == [True, True, False, True, True, False, True]

def test_free_climb_follows_negative_mode_or_drops_to_bias():
    """After a free step, the mode overlapping most with its climb
    direction is followed if negative, else the step uses the bias.

    After a biased step there is nothing to compare, and B's lowest
    mode is climbed.  Eigenvector signs do not matter.

    step  B's two softest modes (eigenvalues)    result
    1     b0, b1 (-1, 0.5)                       free, b0
    2     b1, b0 (-1, 0.5)                       bias: b0 matches the positive mode
    3     b1, b0 (-1, 0.5)                       free, b1 (previous step biased)
    4     p40, p130 (-1, 0.5)                    free, p40 (still the best match)
    5     p100, p190 (-1, 0.5)                   bias: p190 matches p40 better, positive
    6     p100, p190 (0.2, 0.5)                  bias (latch)
    7     p100, p190 (-1, 0.5)                   free, p100 (previous step biased)
    8     p190, p100 (-2, -1)                    free, p100: best match, negative
    9     p190, p100 (-2, -1)                    free, p100 again
    10    p190, p100 (-2, 0.5)                   bias: p100 now positive
    """
    climber = generate_saddleclimb_object()
    climber._step_count = climber.min_directed_steps
    idx = climber.indices
    n = 3 * len(idx)
    climber._pos_i_1D = climber.atoms_initial.positions[idx, :].reshape(-1)
    climber._pos_f_1D = climber.atoms_final.positions[idx, :].reshape(-1)
    pos_1D = 0.5 * (climber._pos_i_1D + climber._pos_f_1D)
    dhat = climber.normalize(climber._pos_f_1D - climber._pos_i_1D)
    basis, _ = LA.qr(dhat.reshape(n, 1), mode='complete')
    b0, b1 = basis[:, 1], basis[:, 2]

    def hessian(first, second, eig1=-1.0, eig2=0.5):
        """First and second eigenvectors given; the rest stiff."""
        vecs, _ = LA.qr(np.column_stack([first, second]), mode='complete')
        eigs = np.concatenate(([eig1, eig2], np.full(n - 2, 5.0)))
        return vecs @ np.diag(eigs) @ vecs.T

    def p(deg):
        """Unit vector at deg from b1 towards b0, across the chord."""
        t = np.radians(deg)
        return np.cos(t) * b1 + np.sin(t) * b0

    steps = [(hessian(b0, b1), b0),
             (hessian(-b1, b0), None),
             (hessian(b1, -b0), b1),
             (hessian(p(40), p(130)), p(40)),
             (hessian(p(100), p(190)), None),
             (hessian(p(100), p(190), eig1=0.2), None),
             (hessian(-p(100), p(190)), p(100)),
             (hessian(p(190), p(100), eig1=-2.0, eig2=-1.0), p(100)),
             (hessian(p(190), -p(100), eig1=-2.0, eig2=-1.0), p(100)),
             (hessian(p(190), p(100), eig1=-2.0), None)]
    for k, (B, climbed) in enumerate(steps):
        g = (-1) ** k * 0.3 * LA.eigh(B)[1][:, 0]
        climber._get_B_opt(B, g, pos_1D)
        assert climber._free == (climbed is not None), k + 1
        if climbed is not None:
            assert_allclose(abs(np.dot(climber._climb_mode, climbed)), 1,
                            atol=1e-10)

def test_tripped_guard_is_a_one_off():
    """A tripped guard biases only the step it trips on.

    The next step, where the guard passes, is free again at once.
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
    eigs[0], eigs[1] = 5.0, -1.0
    B = basis @ np.diag(eigs) @ basis.T

    # Ascent along +w leads away from both ends here, so it trips.
    far = climber._pos_i_1D + 1.3 * chord + 0.5 * w
    mid = climber._pos_i_1D + 0.5 * chord
    free = []
    for pos_1D, g in [(mid, 0.3 * w), (far, 0.5 * dhat + 0.3 * w),
                      (mid, 0.3 * w), (mid, 0.3 * w)]:
        climber._get_B_opt(B, g, pos_1D)
        free.append(climber._free)
    assert free == [True, False, True, True]
