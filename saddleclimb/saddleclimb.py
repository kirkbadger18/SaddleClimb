import sys
import os
import numpy as np
import numpy.linalg as LA
from numpy import matmul as mult
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from scipy.optimize import brentq, minimize
from ase.geometry import find_mic
from pathlib import Path


class SaddleClimb:

    def __init__(
            self: None,
            atoms_initial: Atoms,
            atoms_final: Atoms,
            calculator: Calculator,
            fmax: float = 0.01,
            target_indices: list = None,
            logfile: str = 'climb.log',
            trajfile: str = 'climb.traj',
            interp: str = 'qst',

            delta0: float = 0.1,
            hessian_scale: float = 20,
            min_negative_streak: int = 5,
            maxstep: float = 0.2,

            min_travel: float = 0.1,
            a_max: float = 1,
            max_scaling_halvings: int = 60,
            delta_f: float = 0.05,
            ) -> None:

        self.atoms_initial = atoms_initial
        self.atoms_final = atoms_final
        self.target_indices = target_indices
        self.calculator = calculator
        assert interp in ('linear', 'qst')
        self.interp = interp
        self.delta_f = delta_f
        self.fmax = fmax
        self.maxstep = maxstep
        self.min_travel = min_travel
        self.hessian_scale = hessian_scale
        if not min_negative_streak >= 1:
            raise ValueError('min_negative_streak must be at least 1, '
                             f'got {min_negative_streak}')
        self.min_negative_streak = min_negative_streak
        self.a_max = a_max
        self.max_scaling_halvings = max_scaling_halvings
        self.delta = delta0
        self.logfile = logfile
        self.trajfile = trajfile
        self._restart = False
        self._climbing = True
        self._negative_streak = 0
        self._get_moving_atoms()
        if self.target_indices:
            self._get_sub_target_atoms()
            if not self.sub_target_indices:
                raise ValueError('no atom in target_indices moves '
                                 'between the initial and final states')
        self._bias_atoms = ([self.indices[i] for i in self.sub_target_indices]
                            if self.target_indices else list(self.indices))
        self.hessian = (hessian_scale
                        * np.eye(3*len(self.indices)))
        self._mic_offset = self._get_mic_offset()

    def _get_mic_offset(self):
        """Lattice shift making every pair minimum-image.

        Resolved once from the initial structure.  A pair only changes
        image near the half-cell surface, far outside bonding range, so
        the choice holds for the whole climb.
        """
        pos = self.atoms_initial.positions
        raw = pos[:, None, :] - pos[None, :, :]
        mic, _ = find_mic(raw.reshape(-1, 3), self.atoms_initial.cell,
                          self.atoms_initial.pbc)
        return mic.reshape(raw.shape) - raw

    def _pair_vectors(self, positions):
        """Minimum-image vector between every pair of atoms."""
        return (positions[:, None, :] - positions[None, :, :]
                + self._mic_offset)

    def _pair_distances(self, positions):
        """Minimum-image distance between every pair of atoms."""
        return LA.norm(self._pair_vectors(positions), axis=-1)

    def _get_moving_atoms(self):
        dpos = self.atoms_final.positions - self.atoms_initial.positions
        idx = []
        for i in range(dpos.shape[0]):
            if LA.norm(dpos[i, :]) > 1e-6:
                idx.append(i)
        self.indices = idx.copy()

    def _get_sub_target_atoms(self):
        sub_indices = []
        for i in range(len(self.indices)):
            if self.indices[i] in self.target_indices:
                sub_indices.append(i)
        self.sub_target_indices = sub_indices.copy()

    def _hold_non_targets(self, v_1D):
        """Zero the components of atoms outside ``target_indices``."""
        if self.target_indices:
            for i in range(len(self.indices)):
                if i not in self.sub_target_indices:
                    v_1D[3*i:3*i+3] = 0
        return v_1D

    def _get_bias_vector(self, pos_1D):
        """Bias direction with atoms outside ``target_indices`` held."""
        return self._hold_non_targets(self._get_bias_direction(pos_1D))

    def _get_directed_hessian(self, B, pos_1D):
        """
        B with the bias direction decoupled from the rest.

        The cross terms between the bias and every other direction are
        nulled; no curvature is changed, so the bias keeps its own
        curvature u^T B u whatever its sign.  Returns the decoupled
        Hessian and the unit bias u, an exact eigenvector of it.
        """
        first_column = self._get_bias_vector(pos_1D)
        new_basis, _ = LA.qr(first_column.reshape(-1, 1), mode='complete')
        B_transformed = mult(new_basis.T, mult(B, new_basis))
        B_transformed[1:, 0], B_transformed[0, 1:] = 0, 0
        B_new = mult(new_basis, mult(B_transformed, new_basis.T))
        return B_new, new_basis[:, 0]

    def _is_climbing(self, vmax, g, dxi, dxf):
        """False when the ascent direction leads away from both ends."""
        ascent_dir = vmax if np.dot(g, vmax) > 0 else -vmax
        return not (np.dot(ascent_dir, dxi) < 0
                    and np.dot(ascent_dir, dxf) < 0)

    def _update_negative_streak(self, B):
        """Count consecutive Hessian updates whose lowest mode is negative."""
        if LA.eigvalsh(B)[0] < 0:
            self._negative_streak += 1
        else:
            self._negative_streak = 0

    def _is_directed(self):
        """Steer by the bias until B's negative mode has persisted."""
        return self._negative_streak < self.min_negative_streak

    def _get_B_opt(self, B, g, pos_1D):
        """
        Hessian handed to P-RFO, and the mode it climbs.

        No eigenvalue signs are changed: P-RFO imposes the saddle shape
        itself by taking the highest root along the climb mode and the
        lowest across it, and it climbs a convex mode only if it sees
        that mode's true, positive curvature.  While directed, the bias
        is decoupled from the rest of B and climbed; once free, the
        lowest mode of B is climbed.  The guard falls back to the bias
        when the free mode leads away from both ends, and clears
        ``_climbing`` if neither climbs.  The climb mode is stored in
        ``_climb_mode``.
        """
        dxi = self._pos_i_1D - pos_1D
        dxf = self._pos_f_1D - pos_1D
        eigs_B, vecs_B = LA.eigh(B)
        directed = self._is_directed()
        if directed:
            B_opt, v = self._get_directed_hessian(B, pos_1D)
        else:
            B_opt, v = B, vecs_B[:, 0]
        self._climbing = self._is_climbing(v, g, dxi, dxf)
        if not self._climbing and not directed:
            B_dir, u = self._get_directed_hessian(B, pos_1D)
            if self._is_climbing(u, g, dxi, dxf):
                B_opt, v, self._climbing = B_dir, u, True
        if not self._climbing:
            B_opt, v = B, vecs_B[:, 0]
        self._climb_mode = v
        return B_opt

    def _get_maxstep(self, dx_1D: np.ndarray) -> float:
        """Displacement of the furthest-moving atom.

        The one distance convention used throughout: every step
        length, floor and travel threshold means "how far the
        atom that moves most moves".  Independent of system size,
        and the same reduction ``Fmax`` uses for forces.
        """
        return LA.norm(dx_1D.reshape(-1, 3), axis=1).max()

    def _get_scaled_climb_step(self, B_opt, g, vmax, a):
        """
        Climb component of the P-RFO step, maximised along ``vmax``.

        When the guard has cleared ``_climbing`` this component is
        nulled rather than descended: climb and descent would otherwise
        be the same mode pulling opposite ways, and what is left
        relaxes everything perpendicular to it.
        """
        if not self._climbing:
            return np.zeros_like(g)
        climb_M = np.array([
            [a**2*mult(vmax.T, mult(B_opt, vmax)), a*mult(vmax.T, g)],
            [a*mult(g.T, vmax), 0]
        ])
        _, svecs_max = LA.eigh(climb_M)
        return (a*svecs_max[0, 1] / svecs_max[1, 1]) * vmax

    def _get_scaled_descend_step(self, B_opt, g, vmin, a):
        """Descent component, minimised in the space spanned by ``vmin``."""
        Ndim = len(g)
        descend_M = np.zeros([Ndim, Ndim])
        descend_M[0:Ndim-1, 0:Ndim-1] = a**2*mult(vmin.T, mult(B_opt, vmin))
        descend_M[-1, 0:Ndim-1] = a*mult(vmin.T, g)
        descend_M[0:Ndim-1, -1] = a*mult(g.T, vmin)
        _, svecs_min = LA.eigh(descend_M)
        smin = (a / svecs_min[-1, 0]) * svecs_min[0:Ndim-1, 0]
        return mult(vmin, smin)

    def _get_pfro_scaling(self, component, radius):
        """
        Pick the RFO scaling ``a`` for one step component, starting from
        the nominal ``self.a_max`` and reducing it only if that
        component would leave its own trust ``radius``.

        Near convergence the nominal step is already well inside the
        radius -- it approaches the Newton step, which shrinks with the
        gradient -- so no search happens and ``a_max`` is returned as is.
        When the radius does bind, the step length shrinks monotonically
        as ``a`` is reduced, so ``a`` is halved until the step is inside
        the radius and the resulting bracket is closed with Brent's
        method.  Searching in log(a) keeps the bracket well conditioned
        when many halvings are needed.  Only linear algebra is involved,
        no force evaluations.
        """
        def excess(log_a):
            return (self._get_maxstep(component(np.exp(log_a)))
                    - radius)

        log_hi = np.log(self.a_max)
        if excess(log_hi) <= 0:
            return self.a_max

        log_lo = log_hi
        bracketed = False
        for _ in range(self.max_scaling_halvings):
            log_lo -= np.log(2)
            if excess(log_lo) <= 0:
                bracketed = True
                break

        if not bracketed:
            return np.exp(log_lo)

        log_a = brentq(excess, log_lo, log_hi, xtol=1e-3)
        return np.exp(log_a)

    def _get_pfro_step(self, B_opt, g, a=None):
        """
        Partitioned RFO step: maximized along ``_climb_mode``, minimized
        in the space across it.  One scaling is solved for the whole
        step against the single trust radius ``maxstep``, and both
        partitions use it.  With an explicit ``a`` the
        search is skipped.  The linear truncation is kept as a
        safety net: the bracket is closed only to ``xtol`` in
        log(a), and for the case where no bracket could be found
        at all.
        """
        vmax = self._climb_mode
        basis, _ = LA.qr(vmax.reshape(-1, 1), mode='complete')
        vmin = basis[:, 1:]

        def climb(scale):
            return self._get_scaled_climb_step(B_opt, g, vmax, scale)

        def descend(scale):
            return self._get_scaled_descend_step(B_opt, g, vmin, scale)

        def total(scale):
            return climb(scale) + descend(scale)

        scale = (a if a is not None
                 else self._get_pfro_scaling(total, self.maxstep))
        step = total(scale)
        stepsize = self._get_maxstep(step)
        if stepsize > self.maxstep:
            step = step * (self.maxstep / stepsize)
        return step

    def _get_step(self, B_opt, g):
        """P-RFO step from the decoupled Hessian."""
        return self._get_pfro_step(B_opt, g)

    def _update_hessian(
            self: None, B_old: np.ndarray,
            dg: np.ndarray, dx: float
            ) -> np.ndarray:
        """
        Hessian update procedure described by:

        """
        eig, vec = LA.eigh(B_old)
        dxT = np.transpose(dx)
        dgT = np.transpose(dg)
        B_abs = np.zeros(np.shape(self.hessian))
        for i in range(len(eig)):
            B_abs += np.abs(eig[i]) * np.outer(vec[:, i], vec[:, i].T)
        dx_square = np.outer(dx, dxT)
        dg_square = np.outer(dg, dgT)
        M = dg_square + mult(B_abs, mult(dx_square, B_abs))
        j = dg - mult(B_old, dx)
        u_term = 1/(mult(dxT, mult(M, dx)))
        u = u_term * mult(M, dx)
        E_a = np.outer(u, j.T)
        E_b = np.outer(j, u.T)
        E_c = mult(E_a, np.outer(dx, u.T))
        E = E_a + E_b - E_c
        B = B_old + E
        return B

    def _initialize_atoms(self: None) -> tuple[Atoms, np.ndarray, np.ndarray]:
        atoms = self.atoms_initial.copy()
        constraints = self.atoms_initial.constraints.copy()
        atoms.set_constraint(constraints)
        atoms.calc = self.calculator
        idx = self.indices.copy()
        B_init = self.hessian.copy()
        return atoms, idx, B_init

    def _initialize_atoms_restart(self: None) -> tuple[Atoms,
                                                       np.ndarray,
                                                       np.ndarray]:
        atoms = self._restart_trajectory.copy()
        constraints = self._restart_trajectory.constraints.copy()
        atoms.set_constraint(constraints)
        atoms.calc = self.calculator
        idx = self.indices.copy()
        B_init = np.array(atoms.info["saddleclimb_hessian"])

        return atoms, idx, B_init

    def _initialize_run(self: None, atoms: Atoms, idx: list):
        traj = Trajectory(self.trajfile, 'w')
        g_init = -self._get_F(atoms)[idx, :].reshape(-1)
        E_init = atoms.calc.results['energy']
        traj.write(atoms)
        Fmax = np.max(np.abs(g_init))
        log_string = self._get_log_string(0, E_init, Fmax)
        self._log(log_string)
        return traj, g_init, E_init

    def _initialize_run_restart(self: None, idx: list):
        traj = Trajectory(self.trajfile, 'a')
        g_tot = -self._restart_trajectory.calc.results['forces']
        g = g_tot[idx, :].reshape(-1).copy()
        E = self._restart_trajectory.calc.results['energy'] + 0
        Fmax = LA.norm(-g.reshape(-1, 3), axis=1).max()
        return traj, g, E, Fmax

    def _get_initial_step(
            self: None, idx: list
            ) -> tuple[np.ndarray, np.ndarray]:
        self._pos_f_1D = self.atoms_final.positions[idx, :].reshape(-1)
        self._pos_i_1D = self.atoms_initial.positions[idx, :].reshape(-1)
        dx_1D = self._hold_non_targets(self._pos_f_1D - self._pos_i_1D)
        # Held atoms are zeroed first, so delta0 is the length
        # of the step actually taken, in the same units as the
        # trust radii.
        dx_1D = self.delta * dx_1D / self._get_maxstep(dx_1D)
        dx = dx_1D.reshape(-1, 3)
        return dx, dx_1D

    def _get_log_string(self, n, E, Fmax):
        n_str = str(n).ljust(20)
        E_str = str(np.round(E, 6)).ljust(20)
        F_str = str(np.round(Fmax, 6)).ljust(20)
        log_string = n_str + E_str + F_str
        return log_string

    def _log(self: None, string: str) -> None:
        with open(self.logfile, 'a') as log:
            log.write(string + '\n')
        sys.stdout.flush()

    def _initialize_logging(self: None):
        n_str = 'Iteration'.ljust(20)
        E_str = 'Energy (eV)'.ljust(20)
        F_str = 'Fmax (eV/A)'.ljust(20)
        if self._restart:
            log_string = 'Restarting:\n' + n_str + E_str + F_str
        else:
            log_string = n_str + E_str + F_str
        climb = Path(self.logfile)
        if climb.exists() and not self._restart:
            os.remove(self.logfile)
        self._log(log_string)

    def _get_F(self, atoms):
        try:
            f = atoms.get_forces()
        except Exception:
            print('could not compute forces')
            raise Exception('forces not able to be computed')
        return f

    def climb(self: None, maxsteps=None) -> None:
        self._initialize_logging()
        if self._restart:
            n = self._restart_trajectory.info['saddleclimb_iterations']
            atoms, idx, B = self._initialize_atoms_restart()
            traj, g, E, Fmax = self._initialize_run_restart(idx)
            self._pos_f_1D = self.atoms_final.positions[idx, :].reshape(-1)
            self._pos_i_1D = self.atoms_initial.positions[idx, :].reshape(-1)
            pos_1D = atoms.positions[idx, :].reshape(-1)
            dxi = self._get_maxstep(self._pos_i_1D - pos_1D)
            self._negative_streak = self._restart_trajectory.info.get(
                'saddleclimb_negative_streak', 0)
            B_opt = self._get_B_opt(B, g, pos_1D)
            dx_1D = self._get_step(B_opt, g)
            dx = dx_1D.reshape(-1, 3)
        else:
            atoms, idx, B = self._initialize_atoms()
            traj, g, E = self._initialize_run(atoms, idx)
            dx, dx_1D = self._get_initial_step(idx)
            Fmax, dxi, n = 1, 0, 0
        while Fmax > self.fmax or dxi < self.min_travel:
            atoms.positions[idx, :] += dx
            pos_1D = atoms.positions[idx, :].reshape(-1)
            dxi = self._get_maxstep(self._pos_i_1D - pos_1D)
            g0 = g
            f = self._get_F(atoms)
            g = -f[idx, :].reshape(-1)
            E = atoms.calc.results['energy']
            dg = g - g0
            Fmax = LA.norm(-g.reshape(-1, 3), axis=1).max()
            B = self._update_hessian(B, dg, dx_1D)
            self._update_negative_streak(B)
            B_opt = self._get_B_opt(B, g, pos_1D)
            dx_1D = self._get_step(B_opt, g)
            dx = dx_1D.reshape(-1, 3)
            n += 1
            log_string = self._get_log_string(n, E, Fmax)
            self._log(log_string)
            atoms.info["saddleclimb_hessian"] = B.tolist()
            atoms.info["saddleclimb_hessian_shape"] = B.shape
            atoms.info['saddleclimb_iterations'] = n + 0
            atoms.info['saddleclimb_negative_streak'] = self._negative_streak
            image = atoms.copy()
            image.calc = SinglePointCalculator(image, energy=E, forces=f)
            traj.write(image)
            if maxsteps and n >= maxsteps:
                break

    def restart_climb(self, restart_trajectory: Atoms):
        assert 'saddleclimb_hessian' in restart_trajectory.info
        self._restart = True
        self._restart_trajectory = restart_trajectory
        self.climb()

    def _get_bias_direction(self, pos_1D: np.ndarray) -> np.ndarray:
        """Direction the climb is biased along, over the moving atoms."""
        if self.interp == 'linear':
            return self._pos_f_1D - self._pos_i_1D
        pos = self.atoms_initial.positions.copy()
        pos[self.indices, :] = pos_1D.reshape(-1, 3)
        p_m = self._get_path_coordinate(pos)
        f_minus = max(p_m - self.delta_f, 0)
        f_plus = min(p_m + self.delta_f, 1)
        return (self._get_qst_positions(pos, f_plus)
                - self._get_qst_positions(pos, f_minus))

    def _get_qst_positions(self, pos, f):
        """
        Moving-atom positions at ``f`` on the QST path through pos.

        Only the bias atoms are fitted to the interpolated distances.
        Every other atom is held where it is in ``pos``, so it still
        shapes the fit through its pairs with the bias atoms but does
        not move.
        """
        pos_f = self.get_positions_from_distances(
            self.get_qst_distances(pos, f), pos, self._bias_atoms)
        return pos_f[self.indices, :].reshape(-1)

    def _get_path_coordinate(self, positions: np.ndarray) -> float:
        """
        Path coordinate p of a structure, eqs. (4) and (5).

        Measured over the bias atoms only, so relaxation of atoms
        outside ``target_indices`` does not move p along the path.
        """
        atoms = self._bias_atoms
        d_R = LA.norm(positions[atoms] - self.atoms_initial.positions[atoms])
        d_P = LA.norm(positions[atoms] - self.atoms_final.positions[atoms])
        return d_R / (d_R + d_P)

    def get_qst_distances(self, positions: np.ndarray,
                          f: float) -> np.ndarray:
        """Interpolated distance matrix on the QST path through `positions`.

        Follows eqs. (4)-(7) of Halgren and Lipscomb, Chem. Phys. Lett.
        49 (1977) 225: the path coordinate p_m of the intermediate
        structure sets the quadratic term, and `f` is the interpolation
        parameter at which the distances are evaluated.
        """
        pos_R = self.atoms_initial.positions
        pos_P = self.atoms_final.positions
        p_m = self._get_path_coordinate(positions)
        r_R = self._pair_distances(pos_R)
        r_P = self._pair_distances(pos_P)
        r_M = self._pair_distances(positions)
        denom = p_m * (1 - p_m)
        if denom == 0:
            gamma = np.zeros_like(r_M)
        else:
            gamma = (r_M - (1 - p_m) * r_R - p_m * r_P) / denom
        return (1 - f) * r_R + f * r_P + gamma * f * (1 - f)

    def get_positions_from_distances(self, r_interp: np.ndarray,
                                     positions_guess: np.ndarray,
                                     indices: list = None) -> np.ndarray:
        """Cartesian positions that best reproduce `r_interp`.

        Minimizes S of eq. (3) of Halgren and Lipscomb, Chem. Phys.
        Lett. 49 (1977) 225, starting from (and weakly tethered to)
        `positions_guess`.  Every pair distance enters S, but only
        atoms in `indices` are free to move; the rest are held at their
        guessed positions, where they still constrain the free atoms
        through their shared pairs.  `indices=None` frees every atom.
        Returns an array shaped like `atoms.positions`.

        Pairs are weighted by 1/r_R^4 + 1/r_P^4 from the two endpoint
        structures, rather than by 1/r_interp^4 as in eq. (3), so the
        weights are fixed by the reaction rather than shifting with the
        current geometry.
        """
        free = (np.arange(len(positions_guess)) if indices is None
                else np.asarray(indices))
        iu = np.triu_indices(len(positions_guess), k=1)
        r_i = r_interp[iu]
        r_R = self._pair_distances(self.atoms_initial.positions)[iu]
        r_P = self._pair_distances(self.atoms_final.positions)[iu]
        w = 1 / r_R**4 + 1 / r_P**4
        x0 = positions_guess[free].reshape(-1)

        def S_and_grad(x):
            pos = positions_guess.copy()
            pos[free] = x.reshape(-1, 3)
            dpos = self._pair_vectors(pos)
            r_c = np.sqrt((dpos**2).sum(-1))
            dr = r_c[iu] - r_i
            dx = x - x0
            S = np.sum(w * dr**2) + 1e-6 * np.sum(dx**2)
            c = np.zeros_like(r_c)
            c[iu] = 2 * w * dr / r_c[iu]
            c += c.T
            grad = (c[:, :, None] * dpos).sum(axis=1)[free].reshape(-1)
            return S, grad + 2e-6 * dx

        res = minimize(S_and_grad, x0, jac=True, method='L-BFGS-B')
        out = positions_guess.copy()
        out[free] = res.x.reshape(-1, 3)
        return out

    def normalize(self: None, v: np.ndarray) -> np.ndarray:
        norm = LA.norm(v)
        return v / norm
