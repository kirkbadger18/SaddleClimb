import numpy as np
import numpy.linalg as LA
from numpy import matmul as mult
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from numpy.polynomial import Polynomial
from scipy.optimize import brentq, minimize
from ase.neighborlist import primitive_neighbor_list
from pathlib import Path


class SaddleClimb:

    def __init__(
            self,
            atoms_initial: Atoms,
            atoms_final: Atoms,
            calculator: Calculator,
            fmax: float = 0.01,
            target_indices: list = None,
            logfile: str = 'climb.log',
            trajfile: str = 'climb.traj',
            interp: str = 'qst',

            hessian_scale: float = 50,
            min_directed_steps: int = 8,
            persistence: int = 3,
            maxstep: float = 0.2,

            r_cut: float = 6,
            min_travel: float = 0.1,
            a_max: float = 1,
            max_scaling_halvings: int = 60,
            delta_f: float = 0.01,
            ) -> None:

        self.atoms_initial = atoms_initial
        # R and P are taken in the frame given: each atom's P position is
        # where it ends up from R, so displacements are plain differences.
        self.atoms_final = atoms_final
        self.target_indices = target_indices
        self.calculator = calculator
        assert interp in ('linear', 'qst')
        self.interp = interp
        self.delta_f = delta_f
        # Molecules keep every pair at full weight, as in eq. (2).
        self.r_cut = r_cut if self.atoms_initial.pbc.any() else np.inf
        self.fmax = fmax
        self.maxstep = maxstep
        self.min_travel = min_travel
        if not min_directed_steps >= 0:
            raise ValueError('min_directed_steps must not be negative, '
                             f'got {min_directed_steps}')
        self.min_directed_steps = min_directed_steps
        if not persistence >= 1:
            raise ValueError('persistence must be at least 1, '
                             f'got {persistence}')
        self.persistence = persistence
        self.a_max = a_max
        self.max_scaling_halvings = max_scaling_halvings
        self.logfile = logfile
        self.trajfile = trajfile
        self._restart = False
        self._free = False
        self._climbing = True
        self._step_count = 0
        self._prev_mode = None
        self._streak = 0
        self._E_initial = None
        self._E_final = None
        self._E_cur = None
        dpos = self.atoms_final.positions - self.atoms_initial.positions
        self.indices = [i for i in range(len(dpos))
                        if LA.norm(dpos[i]) > 1e-6]
        if self.target_indices:
            self.sub_target_indices = [
                i for i, atom in enumerate(self.indices)
                if atom in self.target_indices]
            if not self.sub_target_indices:
                raise ValueError('no atom in target_indices moves '
                                 'between the initial and final states')
        self._bias_atoms = ([self.indices[i] for i in self.sub_target_indices]
                            if self.target_indices else list(self.indices))
        self._pos_i_1D = self.atoms_initial.positions[self.indices].reshape(-1)
        self._pos_f_1D = self.atoms_final.positions[self.indices].reshape(-1)
        self.hessian = hessian_scale * np.eye(3*len(self.indices))

    def _get_image_pairs(self, positions):
        """
        Pairs ``(a, b, T)``: atom a and b's image shifted by lattice vector
        T, within ``r_cut`` in R, P or ``positions`` (Govind et al., Comput.
        Mater. Sci. 28 (2003) 250, eq. 5).  T stays fixed along the path, so
        an atom can see two images of a neighbour.  Pair vector x_b - x_a + T.
        """
        if not self.atoms_initial.pbc.any():
            a, b = np.triu_indices(len(positions), k=1)
            return a, b, np.zeros((len(a), 3))
        cell = np.asarray(self.atoms_initial.cell)
        found = []
        for pos in (self.atoms_initial.positions,
                    self.atoms_final.positions, positions):
            i, j, S = primitive_neighbor_list(
                'ijS', self.atoms_initial.pbc, cell, pos, self.r_cut)
            # (b, a, -T) is the same pair, and a == b is a fixed length.
            keep = i < j
            found.append(np.column_stack([i[keep], j[keep], S[keep]]))
        ijS = np.unique(np.concatenate(found), axis=0)
        return ijS[:, 0], ijS[:, 1], ijS[:, 2:] @ cell

    @staticmethod
    def _image_distances(positions, pairs):
        """Distance of each image pair ``(a, b, T)`` in ``positions``."""
        a, b, T = pairs
        return LA.norm(positions[b] - positions[a] + T, axis=1)

    def _cut_weight(self, r):
        """1/r^4 - 1/r_cut^4 inside ``r_cut``, 0 beyond (eq. 5)."""
        with np.errstate(divide='ignore'):
            return np.where(r < self.r_cut, 1 / r**4 - 1 / self.r_cut**4, 0)

    def _hold_non_targets(self, v_1D):
        """Zero the components of atoms outside ``target_indices``."""
        if self.target_indices:
            for i in range(len(self.indices)):
                if i not in self.sub_target_indices:
                    v_1D[3*i:3*i+3] = 0
        return v_1D

    def _get_path_frame(self, pos_1D):
        """
        Path variable f here, tangent t = dx/df (non-targets held), and a
        function giving the moving-atom positions at any f.  The linear path
        is the chord through the current structure; the QST path is the
        quadratic one through it (see ``_get_qst_positions``).
        """
        if self.interp == 'linear':
            chord = self._hold_non_targets(self._pos_f_1D - self._pos_i_1D)
            f = (pos_1D - self._pos_i_1D) @ chord / (chord @ chord)
            return f, chord, lambda f_star: pos_1D + (f_star - f) * chord
        pos = self.atoms_initial.positions.copy()
        pos[self.indices, :] = pos_1D.reshape(-1, 3)
        p_m = self._get_path_coordinate(pos)
        f_minus = max(p_m - self.delta_f, 0)
        f_plus = min(p_m + self.delta_f, 1)
        x_minus = self._get_qst_positions(pos, f_minus)
        x_plus = self._get_qst_positions(pos, f_plus)
        t = self._hold_non_targets(x_plus - x_minus) / (f_plus - f_minus)
        bias_rows = [self.indices.index(i) for i in self._bias_atoms]

        def path_at(f_star):
            # Start the fit from the straight line along the tangent, which
            # is already close, rather than from the current structure.
            guess = pos.copy()
            guess[self._bias_atoms] += ((f_star - p_m)
                                        * t.reshape(-1, 3)[bias_rows])
            return self._get_qst_positions(pos, f_star, guess)

        return p_m, t, path_at

    def _get_directed_hessian(self, B, bias):
        """
        B with the bias direction's cross terms nulled, so the unit bias u
        is an exact eigenvector keeping its curvature u^T B u whatever its
        sign.  Returns the new Hessian and u.
        """
        new_basis, _ = LA.qr(bias.reshape(-1, 1), mode='complete')
        B_transformed = mult(new_basis.T, mult(B, new_basis))
        B_transformed[1:, 0], B_transformed[0, 1:] = 0, 0
        B_new = mult(new_basis, mult(B_transformed, new_basis.T))
        return B_new, new_basis[:, 0]

    def _is_climbing(self, v, g, dxi, dxf):
        """
        False when the climb mode ``v``, oriented uphill along ``g``, points
        away from both the initial (``dxi``) and final (``dxf``) structure.
        """
        if np.dot(v, g) < 0:
            v = -v
        return not (np.dot(v, dxi) < 0 and np.dot(v, dxf) < 0)

    def _update_streak(self, eigs, vecs):
        """
        Count consecutive steps with a persistent negative lowest mode: it
        follows on when the last lowest mode overlaps most with this one
        (absolute overlaps).  Otherwise the streak restarts at one, or at
        zero without a negative lowest eigenvalue.
        """
        if not eigs[0] < 0:
            self._prev_mode, self._streak = None, 0
            return
        follows = (self._prev_mode is not None
                   and np.argmax(np.abs(vecs.T @ self._prev_mode)) == 0)
        self._streak = self._streak + 1 if follows else 1
        self._prev_mode = vecs[:, 0]

    def _get_B_opt(self, B, g, pos_1D):
        """
        Hessian for P-RFO; sets ``_climb_mode``, ``_free``, ``_climbing``.
        Free (B as is, lowest mode) once past ``min_directed_steps`` with a
        ``persistence``-long streak; else directed along the chord to f*.
        A climb mode leading away from both ends trips the guard.
        """
        dxi = self._pos_i_1D - pos_1D
        dxf = self._pos_f_1D - pos_1D
        eigs_B, vecs_B = LA.eigh(B)
        self._update_streak(eigs_B, vecs_B)
        free = (self._step_count >= self.min_directed_steps
                and self._streak >= self.persistence)
        if free:
            B_opt, v = B, vecs_B[:, 0]
        else:
            B_opt, v = self._get_directed_hessian(
                B, self._get_climb_direction(g, pos_1D))
        self._climbing = self._is_climbing(v, g, dxi, dxf)
        self._climb_mode = v
        self._free = free
        return B_opt

    def _get_maxstep(self, dx_1D: np.ndarray) -> float:
        """Length of the largest per-atom vector: steps, travel and Fmax."""
        return LA.norm(dx_1D.reshape(-1, 3), axis=1).max()

    def _get_scaled_climb_step(self, B_opt, g, vmax, a):
        """
        Climb component of the P-RFO step, maximised along ``vmax``; nulled
        when the guard has cleared ``_climbing``, leaving the descent.
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

    def _get_path_energy_max(self, f, G_f):
        """
        f of the energy maximum on a quintic E(f) with minima at both ends,
        fitted to the end energies and the energy and slope ``G_f`` at f.
        The higher interior maximum, or None if there is none, the energies
        are unknown, or the fit is singular.
        """
        if None in (self._E_initial, self._E_final, self._E_cur):
            return None
        E = self._E_cur - self._E_initial
        E_P = self._E_final - self._E_initial
        powers = np.arange(2, 6)
        A = np.array([
            np.ones(4), powers,
            f**powers, powers * f**(powers - 1)])
        try:
            c = LA.solve(A, np.array([E_P, 0.0, E, G_f]))
        except LA.LinAlgError:
            return None
        if not np.all(np.isfinite(c)):
            return None
        energy = Polynomial(np.concatenate([[0, 0], c]))
        slope = energy.deriv()
        curvature = slope.deriv()
        eps = 1e-6
        best = None
        for r in slope.roots():
            if abs(r.imag) > 1e-9 or not eps < r.real < 1 - eps:
                continue
            r = r.real
            if curvature(r) < 0 and (best is None or energy(r) > energy(best)):
                best = r
        return None if best is None else float(best)

    def _get_climb_direction(self, g, pos_1D):
        """
        Unit chord to the path structure at f*: the fitted energy maximum,
        else the middle of the path, f* = 0.5.  The tangent only when f* is
        where we already are.
        """
        f, t, path_at = self._get_path_frame(pos_1D)
        f_max = self._get_path_energy_max(f, t @ g)
        if f_max is None:
            f_max = 0.5
        dx = self._hold_non_targets(path_at(f_max) - pos_1D)
        norm = LA.norm(dx)
        return t / LA.norm(t) if norm == 0 else dx / norm

    def _get_pfro_scaling(self, component, radius):
        """
        RFO scaling: ``a_max`` unless ``component`` would leave ``radius``.
        Then a is halved until it fits and the bracket is closed by Brent's
        method in log(a); the step shrinks monotonically with a.
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

    def _get_pfro_step(self, B_opt, g):
        """
        Partitioned RFO step: maximised along ``_climb_mode``, minimised
        across it, with one scaling for both against ``maxstep``.  The
        final truncation covers the inexact bracket and no bracket at all.
        """
        vmax = self._climb_mode
        basis, _ = LA.qr(vmax.reshape(-1, 1), mode='complete')
        vmin = basis[:, 1:]

        def total(scale):
            return (self._get_scaled_climb_step(B_opt, g, vmax, scale)
                    + self._get_scaled_descend_step(B_opt, g, vmin, scale))

        step = total(self._get_pfro_scaling(total, self.maxstep))
        stepsize = self._get_maxstep(step)
        if stepsize > self.maxstep:
            step = step * (self.maxstep / stepsize)
        return step

    def _update_hessian(self, B_old, dg, dx):
        """Bofill's TS-BFGS update of ``B_old`` for step dx and change dg."""
        eig, vec = LA.eigh(B_old)
        dxT = np.transpose(dx)
        dgT = np.transpose(dg)
        B_abs = np.zeros_like(B_old)
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

    def _initialize_atoms(self) -> tuple[Atoms, list, np.ndarray]:
        atoms = self.atoms_initial.copy()
        atoms.calc = self.calculator
        return atoms, self.indices.copy(), self.hessian.copy()

    def _initialize_atoms_restart(self) -> tuple[Atoms, list, np.ndarray]:
        atoms = self._restart_trajectory.copy()
        atoms.calc = self.calculator
        B_init = np.array(atoms.info["saddleclimb_hessian"])
        return atoms, self.indices.copy(), B_init

    def _initialize_run(self, atoms: Atoms, idx: list):
        traj = Trajectory(self.trajfile, 'w')
        g_init = -atoms.get_forces()[idx, :].reshape(-1)
        E_init = atoms.calc.results['energy']
        traj.write(atoms)
        self._log(self._get_log_string(0, E_init, self._get_maxstep(g_init)))
        return traj, g_init, E_init

    def _get_energy(self, atoms):
        """Energy of an end structure, one extra calculation."""
        atoms = atoms.copy()
        atoms.calc = self.calculator
        return atoms.get_potential_energy()

    def _initialize_run_restart(self, idx: list):
        traj = Trajectory(self.trajfile, 'a')
        results = self._restart_trajectory.calc.results
        g = -results['forces'][idx, :].reshape(-1)
        return traj, g, results['energy'], self._get_maxstep(g)

    def _get_initial_step(self) -> tuple[np.ndarray, np.ndarray]:
        """
        First step, before any energy is known: the chord to the path
        structure at f = 0.5, cut to ``maxstep`` if it is longer.
        """
        _, _, path_at = self._get_path_frame(self._pos_i_1D)
        chord = self._hold_non_targets(path_at(0.5) - self._pos_i_1D)
        dx_1D = chord * min(1.0, self.maxstep / self._get_maxstep(chord))
        return dx_1D.reshape(-1, 3), dx_1D

    def _get_log_string(self, n, E, Fmax):
        return (str(n).ljust(20) + str(np.round(E, 6)).ljust(20)
                + str(np.round(Fmax, 6)).ljust(20))

    def _log(self, string: str) -> None:
        with open(self.logfile, 'a') as log:
            log.write(string + '\n')

    def _initialize_logging(self):
        header = ('Iteration'.ljust(20) + 'Energy (eV)'.ljust(20)
                  + 'Fmax (eV/A)'.ljust(20))
        if self._restart:
            header = 'Restarting:\n' + header
        else:
            Path(self.logfile).unlink(missing_ok=True)
        self._log(header)

    def climb(self, maxsteps=None) -> None:
        self._initialize_logging()
        if self._restart:
            info = self._restart_trajectory.info
            n = info['saddleclimb_iterations']
            atoms, idx, B = self._initialize_atoms_restart()
            traj, g, E, Fmax = self._initialize_run_restart(idx)
            pos_1D = atoms.positions[idx, :].reshape(-1)
            dxi = self._get_maxstep(self._pos_i_1D - pos_1D)
            self._step_count = n
            self._E_initial = info.get('saddleclimb_E_initial')
            self._E_final = info.get('saddleclimb_E_final')
            if self._E_initial is None or self._E_final is None:
                self._E_initial = self._get_energy(self.atoms_initial)
                self._E_final = self._get_energy(self.atoms_final)
            self._E_cur = E
            prev = info.get('saddleclimb_prev_mode')
            self._prev_mode = None if prev is None else np.array(prev)
            self._streak = info.get('saddleclimb_streak', 0)
            B_opt = self._get_B_opt(B, g, pos_1D)
            dx_1D = self._get_pfro_step(B_opt, g)
            dx = dx_1D.reshape(-1, 3)
        else:
            atoms, idx, B = self._initialize_atoms()
            traj, g, E = self._initialize_run(atoms, idx)
            dx, dx_1D = self._get_initial_step()
            self._E_initial = E
            self._E_final = self._get_energy(self.atoms_final)
            Fmax, dxi, n = 1, 0, 0
        while Fmax > self.fmax or dxi < self.min_travel:
            atoms.positions[idx, :] += dx
            pos_1D = atoms.positions[idx, :].reshape(-1)
            dxi = self._get_maxstep(self._pos_i_1D - pos_1D)
            g0 = g
            f = atoms.get_forces()
            g = -f[idx, :].reshape(-1)
            E = atoms.calc.results['energy']
            Fmax = self._get_maxstep(g)
            B = self._update_hessian(B, g - g0, dx_1D)
            self._step_count = n
            self._E_cur = E
            prev_mode, streak = self._prev_mode, self._streak
            B_opt = self._get_B_opt(B, g, pos_1D)
            dx_1D = self._get_pfro_step(B_opt, g)
            dx = dx_1D.reshape(-1, 3)
            n += 1
            self._log(self._get_log_string(n, E, Fmax))
            atoms.info['saddleclimb_hessian'] = B.tolist()
            atoms.info['saddleclimb_iterations'] = n
            atoms.info['saddleclimb_E_initial'] = float(self._E_initial)
            atoms.info['saddleclimb_E_final'] = float(self._E_final)
            atoms.info['saddleclimb_streak'] = int(streak)
            if prev_mode is None:
                atoms.info.pop('saddleclimb_prev_mode', None)
            else:
                atoms.info['saddleclimb_prev_mode'] = prev_mode.tolist()
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

    def _get_path_coordinate(self, positions: np.ndarray) -> float:
        """
        Path coordinate p = d_R / (d_R + d_P), eqs. (4)-(5), over the bias
        atoms only, so relaxing non-targets does not move p.
        """
        atoms = self._bias_atoms
        d_R = LA.norm(positions[atoms] - self.atoms_initial.positions[atoms])
        d_P = LA.norm(positions[atoms] - self.atoms_final.positions[atoms])
        return d_R / (d_R + d_P)

    def _get_qst_positions(self, pos, f, guess=None):
        """
        Moving-atom positions at f on the QST path through ``pos``
        (Halgren and Lipscomb, Chem. Phys. Lett. 49 (1977) 225, eqs. 4-7),
        each image pair weighted by w(r_f) + w(r_R) + w(r_P).  The fit
        starts from ``guess`` (default ``pos``); only bias atoms move.
        """
        pairs = self._get_image_pairs(pos)
        r_R, r_P, r_M = (self._image_distances(x, pairs) for x in (
            self.atoms_initial.positions, self.atoms_final.positions, pos))
        p_m = self._get_path_coordinate(pos)
        denom = p_m * (1 - p_m)
        gamma = 0 if denom == 0 else (r_M - (1 - p_m) * r_R - p_m * r_P) / denom
        r_f = (1 - f) * r_R + f * r_P + gamma * f * (1 - f)
        w = (self._cut_weight(r_f) + self._cut_weight(r_R)
             + self._cut_weight(r_P))
        pos_f = self._fit_positions(pairs, r_f, w,
                                    pos if guess is None else guess)
        return pos_f[self.indices, :].reshape(-1)

    def _fit_positions(self, pairs, r, w, guess):
        """
        Minimise S = sum w (|x_b - x_a + T| - r)^2 over the bias atoms,
        weakly tethered to ``guess``; every other atom is held there.
        Returns an array shaped like ``guess``.
        """
        free = np.asarray(self._bias_atoms)
        # Pairs between held atoms, or with no weight, add a constant.
        a, b, T = pairs
        keep = (np.isin(a, free) | np.isin(b, free)) & (w > 0)
        a, b, T, r, w = a[keep], b[keep], T[keep], r[keep], w[keep]
        x0 = guess[free].reshape(-1)

        def S_and_grad(x):
            pos = guess.copy()
            pos[free] = x.reshape(-1, 3)
            dpos = pos[b] - pos[a] + T
            r_c = LA.norm(dpos, axis=1)
            dr = r_c - r
            dx = x - x0
            S = np.sum(w * dr**2) + 1e-6 * np.sum(dx**2)
            c = (2 * w * dr / r_c)[:, None] * dpos
            grad = np.zeros_like(pos)
            np.add.at(grad, b, c)
            np.add.at(grad, a, -c)
            return S, grad[free].reshape(-1) + 2e-6 * dx

        res = minimize(S_and_grad, x0, jac=True, method='L-BFGS-B')
        out = guess.copy()
        out[free] = res.x.reshape(-1, 3)
        return out
