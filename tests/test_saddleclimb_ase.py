import numpy.linalg as LA
from numpy.testing import assert_allclose
import pytest
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer

from saddleclimb_ase import SaddleClimb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def relaxed_pair():
    """Relaxed initial and final states for a H hop on Pt(111).

    Module-scoped so the two BFGS relaxations run only once per test session.
    Tests must call .copy() before modifying either object.
    """
    slab = fcc111('Pt', size=(2, 2, 3), vacuum=10.0)
    add_adsorbate(slab, 'H', 1.5, 'fcc')
    slab.set_constraint(FixAtoms(mask=[a.tag > 1 for a in slab]))

    atoms_i = slab.copy()
    atoms_i.calc = EMT()
    BFGS(atoms_i, logfile=None).run(fmax=0.01)

    atoms_f = atoms_i.copy()
    atoms_f.positions[-1, 0] += 1.5
    atoms_f.calc = EMT()
    BFGS(atoms_f, logfile=None).run(fmax=0.01)

    return atoms_i, atoms_f


@pytest.fixture
def sc(relaxed_pair, tmp_path):
    """Fresh SaddleClimb instance with EMT calculator and tmp file paths."""
    atoms_i, atoms_f = relaxed_pair
    atoms = atoms_i.copy()
    atoms.calc = EMT()
    return SaddleClimb(
        atoms, atoms_f,
        logfile=str(tmp_path / 'climb.log'),
        trajectory=None,
    )


@pytest.fixture
def sc_with_traj(relaxed_pair, tmp_path):
    """SaddleClimb instance that writes a trajectory for restart tests."""
    atoms_i, atoms_f = relaxed_pair
    atoms = atoms_i.copy()
    atoms.calc = EMT()
    return SaddleClimb(
        atoms, atoms_f,
        logfile=str(tmp_path / 'climb.log'),
        trajectory=str(tmp_path / 'climb.traj'),
    )


# ---------------------------------------------------------------------------
# __init__ / construction
# ---------------------------------------------------------------------------

class TestInit:

    def test_is_ase_optimizer(self, sc):
        assert isinstance(sc, Optimizer)

    def test_moving_indices_nonempty(self, sc):
        assert len(sc.moving_indices) > 0

    def test_moving_indices_type(self, sc):
        assert isinstance(sc.moving_indices, list)

    def test_moving_indices_only_displaced_atoms(self, relaxed_pair, tmp_path):
        atoms_i, atoms_f = relaxed_pair
        atoms = atoms_i.copy()
        atoms.calc = EMT()
        sc = SaddleClimb(
            atoms, atoms_f,
            logfile=str(tmp_path / 'climb.log'),
            trajectory=None,
        )
        dpos = atoms_f.positions - atoms_i.positions
        expected = [i for i in range(len(atoms_i))
                    if LA.norm(dpos[i]) > 1e-6]
        assert sc.moving_indices == expected

    def test_pos_initial_shape(self, sc):
        expected_len = 3 * len(sc.moving_indices)
        assert sc._pos_initial.shape == (expected_len,)

    def test_pos_final_shape(self, sc):
        expected_len = 3 * len(sc.moving_indices)
        assert sc._pos_final.shape == (expected_len,)

    def test_pos_initial_captured_at_construction(
        self, relaxed_pair, tmp_path
    ):
        """_pos_initial must snapshot the geometry passed in."""
        atoms_i, atoms_f = relaxed_pair
        atoms = atoms_i.copy()
        atoms.calc = EMT()
        original_pos = atoms.positions.copy()
        sc = SaddleClimb(
            atoms, atoms_f,
            logfile=str(tmp_path / 'climb.log'),
            trajectory=None,
        )
        # Mutate atoms after construction
        atoms.positions += 5.0
        idx = sc.moving_indices
        assert_allclose(
            sc._pos_initial,
            original_pos[idx].ravel(),
        )

#     def test_hessian_is_scaled_identity(self, sc):
#         n = 3 * len(sc.moving_indices)
#         assert_allclose(sc._hessian, 100 * np.eye(n))

#     def test_hessian_shape(self, sc):
#         n = 3 * len(sc.moving_indices)
#         assert sc._hessian.shape == (n, n)

#     def test_initialized_flag_false(self, sc):
#         assert sc._initialized is False

#     def test_prev_gradient_none(self, sc):
#         assert sc._prev_gradient is None

#     def test_prev_step_none(self, sc):
#         assert sc._prev_step is None

#     def test_maxstep_default(self, sc):
#         assert sc.maxstep == pytest.approx(0.2)

#     def test_maxstep_custom(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f, maxstep=0.1,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         assert sc.maxstep == pytest.approx(0.1)

#     def test_delta_stored(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f, delta0=0.1,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         assert sc.delta == pytest.approx(0.1)

#     def test_target_indices_stored(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         target = [atoms_i.positions.shape[0] - 1]
#         sc = SaddleClimb(
#             atoms, atoms_f, target_indices=target,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         assert sc.target_indices == target

#     def test_target_moving_indices_computed(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         target = [atoms_i.positions.shape[0] - 1]
#         sc = SaddleClimb(
#             atoms, atoms_f, target_indices=target,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         assert hasattr(sc, 'target_moving_indices')
#         # All entries must be valid indices into moving_indices
#         for i in sc.target_moving_indices:
#             assert 0 <= i < len(sc.moving_indices)

#     def test_logfile_cleared_on_fresh_run(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         logfile = tmp_path / 'climb.log'
#         logfile.write_text('old content\n')
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(logfile),
#             trajectory=None,
#         )
#         sc.run(fmax=0.05, steps=1)  # trigger first log write
#         assert 'old content' not in logfile.read_text()

#     def test_identical_initial_final_gives_empty_moving_indices(
#         self, relaxed_pair, tmp_path
#     ):
#         atoms_i, _ = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_i,  # same geometry as final
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         assert sc.moving_indices == []


# # ---------------------------------------------------------------------------
# # Restart validation
# # ---------------------------------------------------------------------------

# class TestRestartValidation:

#     def test_missing_file_raises_file_not_found(self, relaxed_pair,
# tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         with pytest.raises(FileNotFoundError, match="not found"):
#             SaddleClimb(
#                 atoms, atoms_f,
#                 restart=str(tmp_path / 'nonexistent.traj'),
#                 logfile=str(tmp_path / 'climb.log'),
#                 trajectory=None,
#             )

#     def test_traj_without_hessian_raises_value_error(
#         self, relaxed_pair, tmp_path
#     ):
#         atoms_i, atoms_f = relaxed_pair
#         # Write a plain trajectory with no saddleclimb metadata
#         traj_path = tmp_path / 'plain.traj'
#         traj = Trajectory(str(traj_path), 'w')
#         plain = atoms_i.copy()
#         plain.calc = EMT()
#         plain.get_forces()
#         traj.write(plain)
#         traj.close()

#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         with pytest.raises(ValueError, match="Hessian metadata"):
#             SaddleClimb(
#                 atoms, atoms_f,
#                 restart=str(traj_path),
#                 logfile=str(tmp_path / 'climb.log'),
#                 trajectory=None,
#             )

#     def test_logfile_preserved_on_restart(
#         self, relaxed_pair, tmp_path
#     ):
#         """Restart must not clear the existing logfile."""
#         atoms_i, atoms_f = relaxed_pair
#         logfile = tmp_path / 'climb.log'
#         traj_path = tmp_path / 'climb.traj'

#         # Fresh run to produce a valid restart trajectory
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(logfile),
#             trajectory=str(traj_path),
#         )
#         sc.run(fmax=0.05, steps=3)
#         content_after_run = logfile.read_text()

#         atoms2 = atoms_i.copy()
#         atoms2.calc = EMT()
#         SaddleClimb(
#             atoms2, atoms_f,
#             restart=str(traj_path),
#             logfile=str(logfile),
#             trajectory=None,
#         )
#         assert content_after_run in logfile.read_text()


# # ---------------------------------------------------------------------------
# # _normalize
# # ---------------------------------------------------------------------------

# class TestNormalize:

#     def test_unit_length(self):
#         v = np.array([3.0, 4.0, 0.0])
#         assert LA.norm(SaddleClimb._normalize(v)) == pytest.approx(1.0)

#     def test_direction_preserved(self):
#         v = np.array([1.0, 2.0, 3.0])
#         n = SaddleClimb._normalize(v)
#         assert_allclose(n / n[0], v / v[0])

#     def test_works_for_arbitrary_size(self):
#         v = np.random.rand(12)
#         assert LA.norm(SaddleClimb._normalize(v)) == pytest.approx(1.0)


# # ---------------------------------------------------------------------------
# # _get_moving_atoms
# # ---------------------------------------------------------------------------

# class TestGetMovingAtoms:

#     def test_displaced_atoms_included(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         for i in sc.moving_indices:
#             dpos = atoms_f.positions[i] - atoms_i.positions[i]
#             assert LA.norm(dpos) > 1e-6

#     def test_static_atoms_excluded(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         all_indices = set(range(len(atoms_i)))
#         static = all_indices - set(sc.moving_indices)
#         for i in static:
#             dpos = atoms_f.positions[i] - atoms_i.positions[i]
#             assert LA.norm(dpos) <= 1e-6

#     def test_moving_indices_subset_of_all(self, sc):
#         n_atoms = len(sc.atoms)
#         for i in sc.moving_indices:
#             assert 0 <= i < n_atoms


# # ---------------------------------------------------------------------------
# # _get_target_moving_indices
# # ---------------------------------------------------------------------------

# class TestGetTargetMovingIndices:

#     def test_maps_to_moving_indices_subspace(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         last = atoms_i.positions.shape[0] - 1
#         sc = SaddleClimb(
#             atoms, atoms_f, target_indices=[last],
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         for ti in sc.target_moving_indices:
#             assert sc.moving_indices[ti] in sc.target_indices

#     def test_non_moving_target_excluded(self, relaxed_pair, tmp_path):
#         """A target index for a static atom must not appear in
#         target_moving_indices."""
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         # Find a static atom (not in moving_indices)
#         sc_ref = SaddleClimb(
#             atoms_i.copy(), atoms_f,
#             logfile=str(tmp_path / 'ref.log'),
#             trajectory=None,
#         )
#         all_indices = set(range(len(atoms_i)))
#         static = list(all_indices - set(sc_ref.moving_indices))
#         if not static:
#             pytest.skip("No static atoms in this system")

#         atoms2 = atoms_i.copy()
#         atoms2.calc = EMT()
#         sc = SaddleClimb(
#             atoms2, atoms_f, target_indices=[static[0]],
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         assert sc.target_moving_indices == []


# # ---------------------------------------------------------------------------
# # _get_initial_step
# # ---------------------------------------------------------------------------

# class TestGetInitialStep:

#     def test_magnitude_equals_delta(self, sc):
#         step = sc._get_initial_step()
#         # Magnitude per-atom norm should sum to delta in the
# normalised sense;
#         # overall vector norm equals delta.
#         assert LA.norm(step) == pytest.approx(sc.delta)

#     def test_points_toward_final(self, sc):
#         step = sc._get_initial_step()
#         path = sc._pos_final - sc._pos_initial
#         assert np.dot(step, path) > 0

#     def test_target_indices_zeroes_non_targets(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         last = atoms_i.positions.shape[0] - 1
#         sc = SaddleClimb(
#             atoms, atoms_f, target_indices=[last],
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         step = sc._get_initial_step()
#         for i, midx in enumerate(sc.moving_indices):
#             if midx not in sc.target_indices:
#                 assert_allclose(step[3 * i:3 * i + 3], 0.0)


# # ---------------------------------------------------------------------------
# # _get_step
# # ---------------------------------------------------------------------------

# class TestGetStep:

#     def _simple_sc(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         return SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )

#     def test_output_shape(self, relaxed_pair, tmp_path):
#         sc = self._simple_sc(relaxed_pair, tmp_path)
#         n = len(sc.moving_indices) * 3
#         hessian = sc._hessian.copy()
#         gradient = np.random.rand(n)
#         pos_moving = sc._pos_initial.copy()
#         step = sc._get_step(hessian, gradient, pos_moving)
#         assert step.shape == (n,)

#     def test_step_within_maxstep(self, relaxed_pair, tmp_path):
#         sc = self._simple_sc(relaxed_pair, tmp_path)
#         n = len(sc.moving_indices) * 3
#         gradient = np.random.rand(n)
#         pos_moving = sc._pos_initial.copy()
#         step = sc._get_step(sc._hessian, gradient, pos_moving)
#         for i in range(len(sc.moving_indices)):
#             assert LA.norm(step[3 * i:3 * i + 3]) <= sc.maxstep + 1e-10

#     def test_large_step_is_scaled(self, relaxed_pair, tmp_path):
#         """A near-singular Hessian produces a huge raw step; it must be
#         capped to maxstep."""
#         sc = self._simple_sc(relaxed_pair, tmp_path)
#         n = len(sc.moving_indices) * 3
#         # Near-zero Hessian → huge inverse → huge raw step
#         tiny_hessian = 1e-8 * np.eye(n)
#         gradient = np.ones(n)
#         pos_moving = sc._pos_initial.copy()
#         step = sc._get_step(tiny_hessian, gradient, pos_moving)
#         for i in range(len(sc.moving_indices)):
#             assert LA.norm(step[3 * i:3 * i + 3]) <= sc.maxstep + 1e-10

#     def test_negative_eigenvalue_sets_climbing_true(
#         self, relaxed_pair, tmp_path
#     ):
#         sc = self._simple_sc(relaxed_pair, tmp_path)
#         n = len(sc.moving_indices) * 3
#         # Build a Hessian with a negative first eigenvalue
#         hessian = np.eye(n)
#         hessian[0, 0] = -1.0
#         gradient = np.random.rand(n)
#         pos_moving = sc._pos_initial.copy()
#         sc._get_step(hessian, gradient, pos_moving)
#         assert sc._climbing is True

#     def test_gradient_opposing_both_endpoints_disables_climbing(
#         self, relaxed_pair, tmp_path
#     ):
#         """When the gradient points away from both endpoints the algorithm
#         should switch off directed climbing (_climbing = False)."""
#         sc = self._simple_sc(relaxed_pair, tmp_path)
#         n = len(sc.moving_indices) * 3
#         hessian = np.eye(n)

#         # Place pos_moving past the final state so both displacement vectors
#         # point in the same (negative) direction relative to the path.
#         # Then set gradient to point along +path so it opposes both.
#         path = sc._pos_final - sc._pos_initial
#         pos_moving = sc._pos_final + 0.5 * path   # beyond the final state
#         gradient = path / LA.norm(path)            # points away from both

#         sc._get_step(hessian, gradient, pos_moving)
#         assert sc._climbing is False


# # ---------------------------------------------------------------------------
# # _update_hessian
# # ---------------------------------------------------------------------------

# class TestUpdateHessian:

#     def test_output_shape(self, sc):
#         n = len(sc.moving_indices) * 3
#         hessian = sc._hessian.copy()
#         gradient_change = np.random.rand(n)
#         step = np.random.rand(n) * 0.01
#         result = sc._update_hessian(hessian, gradient_change, step)
#         assert result.shape == (n, n)

#     def test_result_is_symmetric(self, sc):
#         n = len(sc.moving_indices) * 3
#         gradient_change = np.random.rand(n)
#         step = np.random.rand(n) * 0.01
#         result = sc._update_hessian(sc._hessian, gradient_change, step)
#         assert_allclose(result, result.T, atol=1e-12)

#     def test_hessian_changes_after_update(self, sc):
#         n = len(sc.moving_indices) * 3
#         hessian_before = sc._hessian.copy()
#         gradient_change = np.random.rand(n)
#         step = np.random.rand(n) * 0.01
#         result = sc._update_hessian(sc._hessian, gradient_change, step)
#         assert not np.allclose(result, hessian_before)


# # ---------------------------------------------------------------------------
# # initialize
# # ---------------------------------------------------------------------------

# class TestInitialize:

#     def test_hessian_is_100_times_identity(self, sc):
#         n = 3 * len(sc.moving_indices)
#         assert_allclose(sc._hessian, 100 * np.eye(n))

#     def test_initialized_is_false(self, sc):
#         assert sc._initialized is False

#     def test_prev_gradient_is_none(self, sc):
#         assert sc._prev_gradient is None

#     def test_prev_step_is_none(self, sc):
#         assert sc._prev_step is None

#     def test_reinitialize_resets_state(self, sc):
#         """Calling initialize() again must restore the pristine state."""
#         sc._hessian *= 2
#         sc._prev_gradient = np.ones(3)
#         sc._prev_step = np.ones(3)
#         sc._initialized = True
#         sc.initialize()
#         n = 3 * len(sc.moving_indices)
#         assert_allclose(sc._hessian, 100 * np.eye(n))
#         assert sc._initialized is False
#         assert sc._prev_gradient is None
#         assert sc._prev_step is None


# # ---------------------------------------------------------------------------
# # read (restart)
# # ---------------------------------------------------------------------------

# class TestRead:

#     def test_positions_restored(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         traj_path = tmp_path / 'climb.traj'

#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=str(traj_path),
#         )
#         sc.run(fmax=0.05, steps=4)
#         checkpoint_pos = sc.atoms.get_positions().copy()

#         atoms2 = atoms_i.copy()
#         atoms2.calc = EMT()
#         sc2 = SaddleClimb(
#             atoms2, atoms_f,
#             restart=str(traj_path),
#             logfile=str(tmp_path / 'climb2.log'),
#             trajectory=None,
#         )
#         assert_allclose(sc2.atoms.get_positions(), checkpoint_pos)

#     def test_hessian_restored(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         traj_path = tmp_path / 'climb.traj'

#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=str(traj_path),
#         )
#         sc.run(fmax=0.05, steps=4)
#         hessian_at_checkpoint = sc._hessian.copy()

#         atoms2 = atoms_i.copy()
#         atoms2.calc = EMT()
#         sc2 = SaddleClimb(
#             atoms2, atoms_f,
#             restart=str(traj_path),
#             logfile=str(tmp_path / 'climb2.log'),
#             trajectory=None,
#         )
#         assert_allclose(sc2._hessian, hessian_at_checkpoint)

#     def test_initialized_true_after_read(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         traj_path = tmp_path / 'climb.traj'

#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=str(traj_path),
#         )
#         sc.run(fmax=0.05, steps=3)

#         atoms2 = atoms_i.copy()
#         atoms2.calc = EMT()
#         sc2 = SaddleClimb(
#             atoms2, atoms_f,
#             restart=str(traj_path),
#             logfile=str(tmp_path / 'climb2.log'),
#             trajectory=None,
#         )
#         assert sc2._initialized is True

#     def test_nsteps_restored(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         traj_path = tmp_path / 'climb.traj'

#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=str(traj_path),
#         )
#         sc.run(fmax=0.05, steps=4)

#         atoms2 = atoms_i.copy()
#         atoms2.calc = EMT()
#         sc2 = SaddleClimb(
#             atoms2, atoms_f,
#             restart=str(traj_path),
#             logfile=str(tmp_path / 'climb2.log'),
#             trajectory=None,
#         )
#         assert sc2.nsteps > 0


# # ---------------------------------------------------------------------------
# # step
# # ---------------------------------------------------------------------------

# class TestStep:

#     def test_first_step_moves_along_path(self, sc):
#         """The very first step should have a positive projection onto the
#         initial→final path direction."""
#         pos_before = sc.atoms.get_positions().copy()
#         sc.step()
#         pos_after = sc.atoms.get_positions().copy()
#         idx = sc.moving_indices
#         displacement = (pos_after - pos_before)[idx].ravel()
#         path = sc._pos_final - sc._pos_initial
#         assert np.dot(displacement, path) > 0

#     def test_moving_atoms_displaced(self, sc):
#         pos_before = sc.atoms.get_positions().copy()
#         sc.step()
#         pos_after = sc.atoms.get_positions().copy()
#         displacement = LA.norm(
#             (pos_after - pos_before)[sc.moving_indices], axis=1
#         )
#         assert np.any(displacement > 0)

#     def test_static_atoms_unchanged(self, sc):
#         all_indices = set(range(len(sc.atoms)))
#         static = list(all_indices - set(sc.moving_indices))
#         if not static:
#             pytest.skip("No static atoms in this system")
#         pos_before = sc.atoms.get_positions().copy()
#         sc.step()
#         pos_after = sc.atoms.get_positions().copy()
#         assert_allclose(pos_after[static], pos_before[static])

#     def test_hessian_stored_in_atoms_info(self, sc):
#         sc.step()
#         assert 'saddleclimb_hessian' in sc.atoms.info

#     def test_iterations_stored_in_atoms_info(self, sc):
#         sc.step()
#         assert 'saddleclimb_iterations' in sc.atoms.info

#     def test_prev_gradient_set_after_step(self, sc):
#         assert sc._prev_gradient is None
#         sc.step()
#         assert sc._prev_gradient is not None

#     def test_prev_step_set_after_step(self, sc):
#         assert sc._prev_step is None
#         sc.step()
#         assert sc._prev_step is not None

#     def test_hessian_updated_after_second_step(self, sc):
#         sc.step()
#         hessian_after_first = sc._hessian.copy()
#         sc.step()
#         # After the second step the Hessian should be updated
#         assert not np.allclose(sc._hessian, hessian_after_first)

#     def test_restart_step_does_not_retake_initial_path_step(
#         self, relaxed_pair, tmp_path
#     ):
#         """After a restart, the first step must be a quasi-Newton step, not
#         a naive delta-step along the path.  The two trajectories should
#         diverge from the checkpoint."""
#         atoms_i, atoms_f = relaxed_pair
#         traj_path = tmp_path / 'climb.traj'

#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=str(traj_path),
#         )
#         sc.run(fmax=0.05, steps=4)

#         # Fresh run from scratch (would take initial path step)
#         atoms_fresh = atoms_i.copy()
#         atoms_fresh.calc = EMT()
#         sc_fresh = SaddleClimb(
#             atoms_fresh, atoms_f,
#             logfile=str(tmp_path / 'fresh.log'),
#             trajectory=None,
#         )
#         # Both fresh and restart call step once; the positions taken should
#         # differ because fresh takes the tiny delta step while restart takes
#         # a quasi-Newton step.
#         sc_fresh.step()
#         fresh_pos = atoms_fresh.get_positions().copy()

#         atoms_restart = atoms_i.copy()
#         atoms_restart.calc = EMT()
#         sc_restart = SaddleClimb(
#             atoms_restart, atoms_f,
#             restart=str(traj_path),
#             logfile=str(tmp_path / 'restart.log'),
#             trajectory=None,
#         )
#         sc_restart.step()
#         restart_pos = atoms_restart.get_positions().copy()

#         assert not np.allclose(fresh_pos, restart_pos)


# # ---------------------------------------------------------------------------
# # converged
# # ---------------------------------------------------------------------------

# class TestConverged:

#     def test_not_converged_when_forces_too_high(self, sc):
#         sc.fmax = 0.001  # very tight — initial forces will exceed this
#         gradient = sc.optimizable.get_gradient()
#         assert not sc.converged(gradient)

#     def test_not_converged_when_too_close_to_initial(self, sc):
#         """Even with forces below fmax the distance condition must block
#         convergence at the initial geometry."""
#         sc.fmax = 1.0  # loose enough to pass force check immediately
#         gradient = sc.optimizable.get_gradient()
#         # Atoms are still at the initial position (no steps taken)
#         assert not sc.converged(gradient)

#     def test_converged_when_both_conditions_met(self, sc):
#         """Simulate convergence by spoofing positions far from initial and
#         providing a near-zero gradient."""
#         sc.fmax = 1.0
#         # Move the recorded initial reference far away so the distance check
#         # passes without actually running the optimizer
#         sc._pos_initial = sc._pos_initial + 10.0
#         gradient = np.zeros(len(sc.atoms) * 3)
#         assert sc.converged(gradient)


# # ---------------------------------------------------------------------------
# # Integration / end-to-end
# # ---------------------------------------------------------------------------

# class TestEndToEnd:

#     def test_run_moves_atoms_from_initial(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         initial_pos = atoms.get_positions().copy()
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         sc.run(fmax=0.05, steps=10)
#         assert not np.allclose(atoms.get_positions(), initial_pos)

#     def test_restart_continues_from_checkpoint(self, relaxed_pair, tmp_path):
#         """Energy at the end of a restarted run should match a continuous run
#         of the same total number of steps."""
#         atoms_i, atoms_f = relaxed_pair
#         traj_path = tmp_path / 'climb.traj'
#         TOTAL_STEPS = 8

#         # Continuous run
#         atoms_cont = atoms_i.copy()
#         atoms_cont.calc = EMT()
#         sc_cont = SaddleClimb(
#             atoms_cont, atoms_f,
#             logfile=str(tmp_path / 'cont.log'),
#             trajectory=None,
#         )
#         sc_cont.run(fmax=0.05, steps=TOTAL_STEPS)
#         energy_cont = atoms_cont.get_potential_energy()

#         # Split run: first half then restart for second half
#         atoms_split = atoms_i.copy()
#         atoms_split.calc = EMT()
#         sc_split = SaddleClimb(
#             atoms_split, atoms_f,
#             logfile=str(tmp_path / 'split.log'),
#             trajectory=str(traj_path),
#         )
#         sc_split.run(fmax=0.05, steps=TOTAL_STEPS // 2)

#         atoms_split2 = atoms_i.copy()
#         atoms_split2.calc = EMT()
#         sc_split2 = SaddleClimb(
#             atoms_split2, atoms_f,
#             restart=str(traj_path),
#             logfile=str(tmp_path / 'split2.log'),
#             trajectory=None,
#         )
#         sc_split2.run(fmax=0.05, steps=TOTAL_STEPS // 2)
#         energy_split = atoms_split2.get_potential_energy()

#         assert energy_cont == pytest.approx(energy_split, rel=1e-4)

#     def test_run_with_target_indices(self, relaxed_pair, tmp_path):
#         atoms_i, atoms_f = relaxed_pair
#         atoms = atoms_i.copy()
#         atoms.calc = EMT()
#         last = len(atoms_i) - 1
#         sc = SaddleClimb(
#             atoms, atoms_f,
#             target_indices=[last],
#             logfile=str(tmp_path / 'climb.log'),
#             trajectory=None,
#         )
#         # Should run without error
#         sc.run(fmax=0.05, steps=5)
