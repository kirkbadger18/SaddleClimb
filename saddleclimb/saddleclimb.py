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
from pathlib import Path
import copy
from ase.mep.neb import idpp_interpolate, NEB


class SaddleClimb:

    def __init__(
            self: None,
            atoms_initial: Atoms,
            atoms_final: Atoms,
            calculator: Calculator,
            step_method: str = 'pfro',
            interp_method: str = 'lst',
            min_directed_steps: int = 5,
            target_indices: list = None,
            fmax: float = 0.01,
            maxstepsize: float = 0.2,
            a_max: float = 1,
            max_scaling_halvings: int = 60,
            delta0: float = 0.05,
            logfile: str = 'climb.log',
            trajfile: str = 'climb.traj',
            ) -> None:

        self.atoms_initial = atoms_initial
        self.atoms_final = atoms_final
        self.target_indices = target_indices
        self.calculator = calculator
        self.step_method = step_method
        self.interp_method = interp_method
        self.min_directed_steps = min_directed_steps
        self.fmax = fmax
        self.maxstepsize = maxstepsize
        self.a_max = a_max
        self.max_scaling_halvings = max_scaling_halvings
        self.delta = delta0
        self.logfile = logfile
        self.trajfile = trajfile
        self._restart = False
        self._directed = True
        self._get_moving_atoms()
        if self.target_indices:
            self._get_sub_target_atoms()
        self.hessian = 100 * np.eye(3*len(self.indices))

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

    def _get_B_opt(self, B, bias_dx, n):

        eigs_B, vecs_B = LA.eigh(B)
        if (eigs_B[0] > 0 or n <= self.min_directed_steps) and self._directed:
            first_column = bias_dx.copy()
            if self.target_indices:
                for i in range(len(self.indices)):
                    if i not in self.sub_target_indices:
                        first_column[3*i:3*i+3] = 0
            new_basis, _ = LA.qr(first_column.reshape(len(bias_dx), 1),
                                 mode='complete')
            B_transformed = mult(new_basis.T, mult(B, new_basis))
            B_transformed[1:, 0], B_transformed[0, 1:] = 0, 0
            B_new = mult(new_basis, mult(B_transformed, new_basis.T))
            eigs_tmp, vecs_tmp = LA.eigh(B_new)
        else:
            eigs_tmp, vecs_tmp = eigs_B.copy(), vecs_B.copy()
            self._directed = False

        for i, eig in enumerate(eigs_tmp):
            if i == 0:
                eigs_tmp[i] = - np.abs(eig)
            else:
                eigs_tmp[i] = np.abs(eig)
        Dmat = np.diag(eigs_tmp)
        B_opt = mult(vecs_tmp, mult(Dmat, vecs_tmp.T))
        return B_opt

    def _get_maxstep(self, dx_1D: np.ndarray) -> float:
        """Largest single-atom displacement in a flattened step."""
        return LA.norm(dx_1D.reshape(-1, 3), axis=1).max()

    def _get_idpp_bias_vector(
            self,
            initial_atoms: Atoms,
            final_atoms: Atoms,
            num_images: int = 9,
            ) -> np.ndarray:

        images = [initial_atoms.copy() for _ in range(num_images - 1)]
        images.append(final_atoms.copy())
        neb = NEB(images)
        neb.interpolate(mic=False)
        idpp_interpolate(neb, fmax=0.001, mic=False, log=None, traj=None)
        pos_0 = images[0].get_positions()
        pos_1 = images[1].get_positions()
        disp = (pos_1 - pos_0)[self.indices, :].copy()
        bias_vector = self.normalize(disp.reshape(-1))
        return bias_vector

    def _get_lst_bias_vector(
            self,
            initial_atoms: Atoms,
            final_atoms: Atoms,
            fraction: float = 0.001,
            gamma: float = 1e-6,
            maxiter: int = 500,
            ) -> np.ndarray:
        """
        Linear synchronous transit bias direction.

        A single intermediate configuration is placed at ``fraction`` along
        the path -- kappa/p of Eq. (2), taken as a continuous parameter
        since one image is wanted rather than a whole chain -- by
        interpolating every pair distance linearly,

            t_ij = d_ij(initial) + fraction * (d_ij(final) - d_ij(initial)),

        and the Cartesian coordinates that best reproduce those targets are
        found by minimising the objective of Halgren and Lipscomb,
        Chem. Phys. Lett. 49, 225 (1977), Eq. (5),

            S(r) = 1/2 sum_(i!=j) w_ij (d_ij(r) - t_ij)^2
                   + gamma/2 sum_i |r_i - r_lin,i|^2 .

        The weight is

            w_ij = 1/d_ij(initial)^4 + 1/d_ij(final)^4 + 1/d_ij(r)^4 ,

        which differs from the 1/d^4 of the IDPP objective in counting both
        endpoints as well as the current distance.  A pair that is long now
        but short in the final state -- a bond that has to form -- is
        weighted as the bond it becomes rather than as the long contact it
        currently is, which is otherwise suppressed by a factor
        (d_long / d_bond)^4.  Retaining the current distance is the reason
        IDPP uses 1/d^4 in the first place: a pair that becomes short during
        the optimisation is still driven apart.

        The sum is a smooth stand-in for 1/min(d1, df, d)^4: it agrees
        wherever one of the three dominates and exceeds it by at most a
        factor of three when they coincide, while avoiding the discontinuous
        derivative that min() introduces wherever two of the distances
        cross.

        The gamma term is the chord restraint of Eq. (5), which removes
        uniform translation and keeps the result attached to the linear
        path.  Halgren and Lipscomb use 1e-6 a.u.; since the weighted pair
        term has units of length^-2, gamma has units of length^-4 and the
        published value converts as 1e-6 bohr^-4 = 1.27e-5 A^-4, doubled
        here to 2.5e-5 because the anchor above carries a factor 1/2 that
        Eq. (5) does not.

        Because the minimisation starts from the linear interpolation, this
        term is nearly inert at that magnitude -- gamma = 0 and gamma =
        2.5e-5 give indistinguishable directions on every case tested.  It
        matters only if the search is started somewhere else, such as at the
        initial state itself, where the pair term alone has almost no
        gradient for a near-isometric rearrangement (a rigid rotation, or a
        hop in which every d_ij(initial) is close to d_ij(final)) and the
        starting point is no longer supplying the direction.

        The default ``fraction`` is small because the result wanted here is
        a direction rather than a path: the returned vector converges to a
        differential displacement as fraction -> 0, and is stable to about
        0.03 degrees between 1e-3 and 1e-7.  Larger values tilt towards the
        chord -- at 0.5 the direction has already moved 16 to 28 degrees off
        that limit.

        Only ``self.indices`` are optimised, every other atom being held at
        its initial position, so constrained atoms cannot drift.  Distances
        are raw Cartesian distances, matching the convention used by
        ``_get_idpp_bias_vector``; a reaction in which an atom crosses a
        periodic boundary needs endpoints in a consistent unwrapped frame.
        """
        pos_i = initial_atoms.get_positions()
        pos_f = final_atoms.get_positions()
        d_i = initial_atoms.get_all_distances(mic=False)
        d_f = final_atoms.get_all_distances(mic=False)

        target = d_i + fraction * (d_f - d_i)
        with np.errstate(divide='ignore'):
            w_end = 1.0 / d_i ** 4 + 1.0 / d_f ** 4
        # the diagonal is excluded by making every self term contribute zero
        np.fill_diagonal(target, 1.0)
        np.fill_diagonal(w_end, 0.0)

        pos_lin = pos_i + fraction * (pos_f - pos_i)
        idx = np.asarray(self.indices, dtype=int)
        if fraction <= 0.0:
            raise ValueError('fraction must be positive')
        # S is quadratic in fraction, and the relative-reduction test in
        # L-BFGS-B divides by max(|S|, 1), so it silently degrades into an
        # absolute test once S falls below one and the search stops early --
        # at fraction=1e-6 after three iterations.  Scaling S to O(1) keeps
        # the tolerances meaningful; it multiplies both terms alike, so the
        # balance against gamma is untouched.
        scale = 1.0 / fraction ** 2

        def objective(x):
            pos = pos_lin.copy()
            pos[idx] = x.reshape(-1, 3)
            # disp[i, j] = r_j - r_i
            disp = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
            dist = np.sqrt((disp ** 2).sum(-1))
            np.fill_diagonal(dist, 1.0)

            w = w_end + 1.0 / dist ** 4
            dd = dist - target
            S = 0.5 * (w * dd ** 2).sum()
            # dS is d/d(dist) of the summand w * dd**2, not of S itself; the
            # factor 1/2 in S cancels against each pair being counted twice.
            # The second term is the derivative of the 1/dist**4 part of w.
            dS = 2.0 * w * dd - 4.0 * dd ** 2 / dist ** 5

            coef = dS / dist
            np.fill_diagonal(coef, 0.0)
            grad = -(coef[..., np.newaxis] * disp).sum(axis=1)

            offset = pos - pos_lin
            S += 0.5 * gamma * (offset ** 2).sum()
            grad += gamma * offset
            return scale * S, scale * grad[idx].reshape(-1)

        result = minimize(objective, pos_lin[idx].reshape(-1), jac=True,
                          method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-14,
                                   'gtol': 1e-10})
        pos_opt = pos_lin.copy()
        pos_opt[idx] = result.x.reshape(-1, 3)
        disp = (pos_opt - pos_i)[idx, :]
        if LA.norm(disp) < 1e-10:
            raise ValueError('LST produced a null bias direction; the two '
                             'endpoints are identical over the moving atoms')
        return self.normalize(disp.reshape(-1))

    def _get_bias_vector(self, initial_atoms: Atoms,
                         final_atoms: Atoms) -> np.ndarray:
        """Bias direction from whichever interpolation was requested."""
        if self.interp_method == 'idpp':
            return self._get_idpp_bias_vector(initial_atoms, final_atoms)
        if self.interp_method == 'lst':
            return self._get_lst_bias_vector(initial_atoms, final_atoms)
        raise ValueError(f"unknown interp_method {self.interp_method!r}, "
                         "expected 'idpp' or 'lst'")

    def _get_newton_step(self, B_opt, g):

        inv_B_temp = LA.inv(B_opt)
        dx_1D = -mult(inv_B_temp, g)
        maxstep = self._get_maxstep(dx_1D)
        if maxstep > self.maxstepsize:
            dx_1D *= self.maxstepsize / maxstep

        return dx_1D

    def _get_scaled_pfro_step(self, B_opt, g, vmin, vmax, a):
        """
        Partitioned RFO step for a given RFO scaling parameter ``a``.

        The step is maximised along ``vmax`` and minimised in the space
        spanned by ``vmin``.  ``a`` acts as a trust radius control: the
        step tends to zero as a -> 0 and to the full Newton/RFO step as
        a -> inf.
        """
        Ndim = len(g)
        climb_M = np.array([
            [a**2*mult(vmax.T, mult(B_opt, vmax)), a*mult(vmax.T, g)],
            [a*mult(g.T, vmax), 0]
        ])
        descend_M = np.zeros([Ndim, Ndim])
        descend_M[0:Ndim-1, 0:Ndim-1] = a**2*mult(vmin.T, mult(B_opt, vmin))
        descend_M[-1, 0:Ndim-1] = a*mult(vmin.T, g)
        descend_M[0:Ndim-1, -1] = a*mult(g.T, vmin)
        _, svecs_min = LA.eigh(descend_M)
        _, svecs_max = LA.eigh(climb_M)
        smax = a*svecs_max[0, 1] / svecs_max[1, 1]
        smin = (a / svecs_min[-1, 0]) * svecs_min[0:Ndim-1, 0]
        step = smax * vmax + mult(vmin, smin)
        return step

    def _get_pfro_scaling(self, B_opt, g, vmin, vmax):
        """
        Pick the RFO scaling ``a``, starting from the nominal
        ``self.a_max`` and reducing it only if that step would leave the
        trust radius ``self.maxstepsize``.

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
            step = self._get_scaled_pfro_step(B_opt, g, vmin, vmax,
                                              np.exp(log_a))
            return self._get_maxstep(step) - self.maxstepsize

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
        Partitioned RFO step.  With ``a=None`` the nominal scaling
        ``self.a_max`` is used, reduced only far enough to keep the step
        within the trust radius; passing an explicit ``a`` skips that
        entirely.  The linear truncation is kept as a safety net for the
        case where no bracket could be found.
        """
        _, vecs = LA.eigh(B_opt)
        vmin, vmax = vecs[:, 1:], vecs[:, 0]
        if a is None:
            a = self._get_pfro_scaling(B_opt, g, vmin, vmax)
        step = self._get_scaled_pfro_step(B_opt, g, vmin, vmax, a)
        maxstep = self._get_maxstep(step)
        if maxstep > self.maxstepsize:
            step *= self.maxstepsize / maxstep
        return step

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
        atoms.calc = copy.deepcopy(self.calculator)
        idx = self.indices.copy()
        B_init = self.hessian.copy()
        return atoms, idx, B_init

    def _initialize_atoms_restart(self: None) -> tuple[Atoms,
                                                       np.ndarray,
                                                       np.ndarray]:
        atoms = self._restart_trajectory.copy()
        constraints = self._restart_trajectory.constraints.copy()
        atoms.set_constraint(constraints)
        atoms.calc = copy.deepcopy(self.calculator)
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
        v = self._get_bias_vector(self.atoms_initial, self.atoms_final)
        dx_1D = self.delta * v / self._get_maxstep(v)
        self._pos_f_1D = self.atoms_final.positions[idx, :].reshape(-1)
        self._pos_i_1D = self.atoms_initial.positions[idx, :].reshape(-1)
        if self.target_indices:
            for i in range(len(self.indices)):
                if i not in self.sub_target_indices:
                    dx_1D[3*i:3*i+3] = 0
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
            self._directed = self._restart_trajectory.info['directed']
            atoms, idx, B = self._initialize_atoms_restart()
            traj, g, E, Fmax = self._initialize_run_restart(idx)
            self._pos_f_1D = self.atoms_final.positions[idx, :].reshape(-1)
            self._pos_i_1D = self.atoms_initial.positions[idx, :].reshape(-1)
            pos_1D = atoms.positions[idx, :].reshape(-1)
            dxi = LA.norm(self._pos_i_1D - pos_1D)
            bias_dx = self._get_bias_vector(atoms, self.atoms_final)
            B_opt = self._get_B_opt(B, bias_dx, n-1)
            dx_1D = self._get_pfro_step(B_opt, g)
            dx = dx_1D.reshape(-1, 3)
        else:
            atoms, idx, B = self._initialize_atoms()
            traj, g, E = self._initialize_run(atoms, idx)
            dx, dx_1D = self._get_initial_step(idx)
            Fmax, dxi, n = 1, 0, 0
        while Fmax > self.fmax or dxi < 0.1:
            atoms.positions[idx, :] += dx
            pos_1D = atoms.positions[idx, :].reshape(-1)
            dxi = LA.norm(self._pos_i_1D - pos_1D)
            g0 = g
            f = self._get_F(atoms)
            g = -f[idx, :].reshape(-1)
            E = atoms.calc.results['energy']
            dg = g - g0
            Fmax = LA.norm(-g.reshape(-1, 3), axis=1).max()

            B = self._update_hessian(B, dg, dx_1D)
            if self._directed:
                bias_dx = self._get_bias_vector(atoms, self.atoms_final)
            B_opt = self._get_B_opt(B, bias_dx, n)
            if self.step_method == 'pfro':
                dx_1D = self._get_pfro_step(B_opt, g)
            elif self.step_method == 'newton':
                dx_1D = self._get_newton_step(B_opt, g)
            dx = dx_1D.reshape(-1, 3)
            n += 1
            log_string = self._get_log_string(n, E, Fmax)
            self._log(log_string)
            atoms.info["saddleclimb_hessian"] = B.tolist()
            atoms.info["saddleclimb_hessian_shape"] = B.shape
            atoms.info['saddleclimb_iterations'] = n + 0
            atoms.info['directed'] = self._directed
            # write a snapshot: some calculators (e.g. MACE) invalidate their
            # results when atoms.info changes, which would drop the energy
            image = atoms.copy()
            image.calc = SinglePointCalculator(image, energy=E, forces=f)
            traj.write(image)
            if maxsteps and n >= maxsteps:
                break

    def restart_climb(self, restart_trajectory: Atoms):
        assert 'saddleclimb_hessian' in restart_trajectory.info
        self._restart = True
        self._restart_trajectory = copy.deepcopy(restart_trajectory)
        self.climb()

    def normalize(self: None, v: np.ndarray) -> np.ndarray:
        norm = LA.norm(v)
        return v / norm
