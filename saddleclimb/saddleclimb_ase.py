import os
import numpy as np
import numpy.linalg as LA
from numpy import matmul as mult
from pathlib import Path
from typing import IO

from ase.atoms import Atoms
from ase.optimize.optimize import Optimizer


class SaddleClimb(Optimizer):
    """
    SaddleClimb: Path-directed quasi-Newton first-order saddle-point search.

    Inherits from the ASE Optimizer base class for full compatibility with
    the ASE ecosystem (calculators, constraints, observers, trajectory).

    The algorithm initialises an approximate Hessian and takes a small
    initial step toward the final state.  Subsequent steps use a TS-BFGS
    Hessian update.  When the Hessian has no negative eigenvalue the
    initial→final direction is enforced via a congruence transformation
    (QR projection) so the climb stays on course.  Once the first
    eigenvalue turns negative, the method switches to unmodified
    minimum-mode following.

    Parameters
    ----------
    atoms_initial : Atoms
        Working atoms object *with calculator attached*, positioned at the
        relaxed initial state (reactant).  Its positions are recorded as
        the reference initial geometry.
    atoms_final : Atoms
        Relaxed final state (product).  Defines the climbing direction.
    target_indices : list of int, optional
        Subset of atom indices whose displacement defines the climbing
        direction.  When provided only these atoms steer the path; all
        moving atoms are still relaxed.
    maxstep : float
        Maximum displacement of any single atom per step (Å).  Default 0.2.
    delta0 : float
        Size of the initial step along the reaction path (Å).  Default 0.05.
    restart : str or Path, optional
        Trajectory file from a previous run.  The last frame (which must
        carry ``saddleclimb_hessian`` metadata) is used to resume the search.
    logfile : str, Path, or IO, optional
        Log file.  Use ``'-'`` for stdout.  Default ``'climb.log'``.
    trajectory : str or Path, optional
        Trajectory file for intermediate structures.  Default ``'climb.traj'``.
    append_trajectory : bool
        Append to an existing trajectory file instead of overwriting it.
        Default ``False``.
    """

    def __init__(
        self,
        atoms_initial: Atoms,
        atoms_final: Atoms,
        target_indices: list = None,
        maxstep: float | None = None,
        delta0: float = 0.05,
        restart=None,
        logfile: IO | str | Path = 'climb.log',
        trajectory: str | Path = 'climb.traj',
        append_trajectory: bool = False,
        **kwargs,
    ):
        self.atoms_final = atoms_final
        self.target_indices = target_indices
        self.maxstep = maxstep or self.defaults['maxstep']
        self.delta = delta0

        self._get_moving_atoms(atoms_initial)
        if self.target_indices:
            self._get_target_moving_indices()

        self._pos_initial = (
            atoms_initial.positions[self.moving_indices].reshape(-1).copy()
        )
        self._pos_final = (
            atoms_final.positions[self.moving_indices].reshape(-1).copy()
        )

        # If restart is requested the file must exist and carry Hessian
        # metadata.
        # ASE's base class silently falls back to initialize() when the file is
        # absent; we treat that as a hard error so a bad restart path never
        # causes a silent re-initialisation from a scaled identity Hessian.
        if restart is not None and not Path(restart).is_file():
            raise FileNotFoundError(
                f"Restart file '{restart}' not found. "
                "SaddleClimb will not fall back to a scaled identity Hessian. "
                "Set restart=None to start a fresh run."
            )

        # Clear logfile before the base class opens it in append mode
        if restart is None and isinstance(logfile, (str, Path)):
            if Path(logfile).exists():
                os.remove(logfile)

        super().__init__(
            atoms=atoms_initial,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # ASE Optimizer interface
    # ------------------------------------------------------------------

    def initialize(self):
        """Set up fresh state (called automatically when restart=None)."""
        n = 3 * len(self.moving_indices)
        self._hessian = 100 * np.eye(n)
        self._prev_gradient = None  # gradient at previous step position
        self._prev_step = None      # previous step vector (flattened)
        self._initialized = False   # False → take initial path step next

    def read(self):
        """Load state from the last frame of the restart trajectory."""
        from ase.io import read as ase_read
        frame = ase_read(str(self.restart), index=-1)
        if 'saddleclimb_hessian' not in frame.info:
            raise ValueError(
                f"Restart file '{self.restart}' does not contain "
                "SaddleClimb Hessian metadata.  Make sure the "
                "trajectory was written by SaddleClimb."
            )
        self.atoms.set_positions(frame.get_positions())
        self._hessian = np.array(frame.info['saddleclimb_hessian'])
        self.nsteps = frame.info.get('saddleclimb_iterations', 0)
        self._prev_gradient = None
        self._prev_step = None
        self._initialized = True  # skip initial path step; use quasi-Newton

    def step(self, gradient=None):
        """Compute and apply one SaddleClimb step."""
        if gradient is None:
            gradient = self.optimizable.get_gradient()

        moving_idx = self.moving_indices
        gradient_moving = gradient.reshape(-1, 3)[moving_idx].ravel()

        if not self._initialized:
            # First step of a fresh run: move delta along the reaction path
            step = self._get_initial_step()
            self._initialized = True

        elif self._prev_step is None:
            # First step after restart: quasi-Newton, no Hessian update
            pos_moving = (
                self.optimizable.get_x().reshape(-1, 3)[moving_idx].ravel()
            )
            step = self._get_step(self._hessian, gradient_moving, pos_moving)

        else:
            # Normal: TS-BFGS Hessian update then quasi-Newton step
            gradient_change = gradient_moving - self._prev_gradient
            self._hessian = self._update_hessian(
                self._hessian, gradient_change, self._prev_step
            )
            pos_moving = (
                self.optimizable.get_x().reshape(-1, 3)[moving_idx].ravel()
            )
            step = self._get_step(self._hessian, gradient_moving, pos_moving)

        self._prev_gradient = gradient_moving
        self._prev_step = step

        # Embed Hessian in atoms.info so trajectory frames carry restart data
        self.atoms.info['saddleclimb_hessian'] = self._hessian.tolist()
        self.atoms.info['saddleclimb_hessian_shape'] = list(
            self._hessian.shape
        )
        self.atoms.info['saddleclimb_iterations'] = self.nsteps

        # Apply step to moving atoms only
        positions = self.optimizable.get_x().reshape(-1, 3)
        positions[moving_idx] += step.reshape(-1, 3)
        self.optimizable.set_x(positions.ravel())

    def converged(self, gradient):
        """Converged when max force < fmax AND >= 0.5 Å from initial."""
        if not super().converged(gradient):
            return False
        moving_idx = self.moving_indices
        pos_moving = (
            self.optimizable.get_x().reshape(-1, 3)[moving_idx].ravel()
        )
        return LA.norm(self._pos_initial - pos_moving) >= 0.5

    # ------------------------------------------------------------------
    # Algorithm internals (identical logic to the standalone SaddleClimb)
    # ------------------------------------------------------------------

    def _get_moving_atoms(self, atoms_initial):
        dpos = self.atoms_final.positions - atoms_initial.positions
        self.moving_indices = [
            i for i in range(dpos.shape[0]) if LA.norm(dpos[i]) > 1e-6
        ]

    def _get_target_moving_indices(self):
        self.target_moving_indices = [
            i for i, atom_idx in enumerate(self.moving_indices)
            if atom_idx in self.target_indices
        ]

    def _get_initial_step(self):
        """Small step of size delta along the normalised initial→final path."""
        step = self.delta * self._normalize(
            self._pos_final - self._pos_initial
        )
        if self.target_indices:
            for i in range(len(self.moving_indices)):
                if i not in self.target_moving_indices:
                    step[3 * i:3 * i + 3] = 0
        return step

    def _get_step(self, hessian, gradient, pos_moving):
        """Compute the next quasi-Newton step with Hessian mode selection."""
        disp_to_initial = self._pos_initial - pos_moving
        disp_to_final = self._pos_final - pos_moving
        path_direction = self._pos_final - self._pos_initial
        eigenvalues, eigenvectors = LA.eigh(hessian)

        self._climbing = True
        if np.dot(gradient, disp_to_initial) < 0 and \
                np.dot(gradient, disp_to_final) < 0:
            # Gradient points away from both endpoints → saddle converging
            mod_eigenvalues = eigenvalues.copy()
            mod_eigenvectors = eigenvectors.copy()
            self._climbing = False
        elif eigenvalues[0] < 0:
            # Negative eigenvalue present → use eigendecomposition directly
            mod_eigenvalues = eigenvalues.copy()
            mod_eigenvectors = eigenvectors.copy()
        else:
            # Project out couplings to initial→final direction via QR
            climb_direction = path_direction.copy()
            if self.target_indices:
                for i in range(len(self.moving_indices)):
                    if i not in self.target_moving_indices:
                        climb_direction[3 * i:3 * i + 3] = 0
            qr_basis, _ = LA.qr(
                climb_direction.reshape(len(path_direction), 1),
                mode='complete',
            )
            hessian_in_basis = mult(qr_basis.T, mult(hessian, qr_basis))
            hessian_in_basis[1:, 0] = 0
            hessian_in_basis[0, 1:] = 0
            hessian_qr = mult(qr_basis, mult(hessian_in_basis, qr_basis.T))
            mod_eigenvalues, mod_eigenvectors = LA.eigh(hessian_qr)

        # Negate first eigenvalue to climb; absolute value for the rest
        for i, eig in enumerate(mod_eigenvalues):
            if i == 0 and self._climbing:
                mod_eigenvalues[i] = -np.abs(eig)
            else:
                mod_eigenvalues[i] = np.abs(eig)

        modified_hessian = mult(
            mod_eigenvectors,
            mult(np.diag(mod_eigenvalues), mod_eigenvectors.T),
        )
        step = -mult(LA.inv(modified_hessian), gradient)

        # Limit the maximum per-atom displacement
        largest_displacement = max(
            LA.norm(step[3 * i:3 * i + 3])
            for i in range(len(self.moving_indices))
        )
        if largest_displacement > self.maxstep:
            step *= self.maxstep / largest_displacement

        return step

    def _update_hessian(self, hessian, gradient_change, step):
        """TS-BFGS approximate Hessian update."""
        eigenvalues, eigenvectors = LA.eigh(hessian)
        abs_hessian = sum(
            np.abs(eigenvalues[i]) * np.outer(
                eigenvectors[:, i], eigenvectors[:, i]
            )
            for i in range(len(eigenvalues))
        )
        M = (np.outer(gradient_change, gradient_change)
             + mult(abs_hessian, mult(np.outer(step, step), abs_hessian)))
        hessian_residual = gradient_change - mult(hessian, step)
        u = mult(M, step) / mult(step, mult(M, step))
        E_a = np.outer(u, hessian_residual)
        E_b = np.outer(hessian_residual, u)
        E_c = mult(E_a, np.outer(step, u))
        return hessian + E_a + E_b - E_c

    @staticmethod
    def _normalize(v):
        return v / LA.norm(v)
