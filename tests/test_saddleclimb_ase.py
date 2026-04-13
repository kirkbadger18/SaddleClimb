import numpy as np
import numpy.linalg as LA
import pytest
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer

from saddleclimb_ase import SaddleClimb


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_pair():
    """Fresh relaxed initial/final pair for a H hop on Pt(111)."""
    slab = fcc111('Pt', size=(2, 2, 3), vacuum=10.0)
    add_adsorbate(slab, 'H', 1.5, 'fcc')
    slab.set_constraint(FixAtoms(mask=[a.tag > 1 for a in slab]))
    atoms_i = slab.copy()
    atoms_i.calc = EMT()
    BFGS(atoms_i, logfile=None).run(fmax=0.05)
    atoms_f = atoms_i.copy()
    atoms_f.positions[-1, 0] += 1.5
    atoms_f.calc = EMT()
    BFGS(atoms_f, logfile=None).run(fmax=0.05)
    return atoms_i, atoms_f


@pytest.fixture
def pair():
    return _make_pair()


@pytest.fixture
def sc(pair, tmp_path):
    atoms_i, atoms_f = pair
    atoms = atoms_i.copy()
    atoms.calc = EMT()
    return SaddleClimb(
        atoms, atoms_f,
        logfile=str(tmp_path / 'climb.log'),
        trajectory=None,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_is_ase_optimizer(sc):
    assert isinstance(sc, Optimizer)


def test_moving_indices_nonempty(sc):
    assert len(sc.moving_indices) > 0


def test_hessian_is_scaled_identity(sc):
    n = 3 * len(sc.moving_indices)
    assert np.allclose(sc._hessian, 100 * np.eye(n))


def test_initialized_false_on_fresh(sc):
    assert sc._initialized is False


def test_missing_restart_raises(pair, tmp_path):
    atoms_i, atoms_f = pair
    atoms = atoms_i.copy()
    atoms.calc = EMT()
    with pytest.raises(FileNotFoundError, match="not found"):
        SaddleClimb(
            atoms, atoms_f,
            restart=str(tmp_path / 'nonexistent.traj'),
            logfile=str(tmp_path / 'climb.log'),
            trajectory=None,
        )


def test_traj_without_hessian_raises(pair, tmp_path):
    atoms_i, atoms_f = pair
    traj_path = tmp_path / 'plain.traj'
    plain = atoms_i.copy()
    plain.calc = EMT()
    plain.get_forces()
    traj = Trajectory(str(traj_path), 'w')
    traj.write(plain)
    traj.close()

    atoms = atoms_i.copy()
    atoms.calc = EMT()
    with pytest.raises(ValueError, match="Hessian metadata"):
        SaddleClimb(
            atoms, atoms_f,
            restart=str(traj_path),
            logfile=str(tmp_path / 'climb.log'),
            trajectory=None,
        )


# ---------------------------------------------------------------------------
# Single step
# ---------------------------------------------------------------------------

def test_step_moves_atoms(sc):
    atoms = sc.atoms
    pos_before = atoms.get_positions().copy()
    sc.step()
    assert not np.allclose(atoms.get_positions(), pos_before)


def test_step_writes_hessian_to_atoms_info(sc):
    sc.step()
    assert 'saddleclimb_hessian' in sc.atoms.info


def test_second_step_updates_hessian(sc):
    sc.step()
    h_after_first = sc._hessian.copy()
    sc.step()
    assert not np.allclose(sc._hessian, h_after_first)


# ---------------------------------------------------------------------------
# Restart (manual trajectory write so we control the frame content)
# ---------------------------------------------------------------------------

def test_restart_reads_positions_and_hessian(pair, tmp_path):
    atoms_i, atoms_f = pair
    traj_path = tmp_path / 'climb.traj'

    # Build a small trajectory manually so hessian metadata is guaranteed
    # in every frame (step() sets atoms.info before we call traj.write).
    atoms = atoms_i.copy()
    atoms.calc = EMT()
    sc = SaddleClimb(
        atoms, atoms_f,
        logfile=str(tmp_path / 'climb.log'),
        trajectory=None,
    )
    traj = Trajectory(str(traj_path), 'w')
    for _ in range(3):
        sc.step()
        traj.write(atoms)
    traj.close()
    checkpoint_pos = atoms.get_positions().copy()
    checkpoint_hessian = sc._hessian.copy()

    # Restart should restore positions and Hessian from the last frame
    atoms2 = atoms_i.copy()
    atoms2.calc = EMT()
    sc2 = SaddleClimb(
        atoms2, atoms_f,
        restart=str(traj_path),
        logfile=str(tmp_path / 'climb2.log'),
        trajectory=None,
    )
    assert np.allclose(sc2.atoms.get_positions(), checkpoint_pos)
    assert np.allclose(sc2._hessian, checkpoint_hessian)
    assert sc2._initialized is True


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_run_reaches_saddle(pair, tmp_path):
    atoms_i, atoms_f = pair
    atoms = atoms_i.copy()
    atoms.calc = EMT()
    sc = SaddleClimb(
        atoms, atoms_f,
        logfile=str(tmp_path / 'climb.log'),
        trajectory=None,
    )
    sc.run(fmax=0.01)
    pos_initial = atoms_i.positions[sc.moving_indices].ravel()
    pos_saddle = atoms.get_positions()[sc.moving_indices].ravel()
    assert LA.norm(pos_initial - pos_saddle) >= 0.5
