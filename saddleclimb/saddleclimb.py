import sys
import os
import numpy as np
import numpy.linalg as LA
from numpy import matmul as mult
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.data import covalent_radii
from ase.units import Bohr
from ase.geometry import get_distances
from ase.geometry import find_mic
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from scipy.optimize import minimize, brentq
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
            interp_method: str = 'bond',
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

    def _get_lst_bias_vector(self, initial_atoms, final_atoms, f=0.01,
                             lst_elements=('C', 'H', 'N', 'O'),
                             cutoff=2):
        """Bias direction from the linear synchronous transit path.

        Follows Halgren and Lipscomb, Chem. Phys. Lett. 49 (1977) 225.
        The LST structure at interpolation parameter ``f`` is the one whose
        interatomic distances best match the linearly interpolated distances
        r_ab(i) = (1-f) r_ab(R) + f r_ab(P), found by minimizing

            S = sum_{a>b} [r_ab(c) - r_ab(i)]^2 / r_ab(i)^4
                + 1e-6 sum_a |x_a(c) - x_a(i)|^2

        with all quantities in atomic units.  The weight is taken as
        1/min(r_ab(R), r_ab(P))^4 rather than the paper's 1/r_ab(i)^4, so a
        pair that is bonded at either end of the path keeps a large weight
        for every f.  A forming bond is therefore reproduced as closely as a
        breaking one, instead of being ignored until late in the path.

        The reference x_a(i) of the second term is ``initial_atoms`` itself,
        not the linearly interpolated coordinates the paper uses.  The fit is
        underdetermined once the distance sum is restricted, so that term is
        what fixes the leftover degrees of freedom: referencing the straight
        line makes every unconstrained coordinate drift along it, while
        referencing the current image asks instead for the smallest move that
        reproduces the target distances.  Only the moving atoms are varied.

        The distance sum is restricted to pairs in which *both* atoms are of
        an element in ``lst_elements`` and which are closer than ``cutoff``
        (in Angstrom) at one end of the path or the other, so the fit is
        driven by the bonds that make and break rather than by the substrate
        or by distant contacts.  Only the resulting direction is used, so
        the structure is not meant to reproduce the final geometry
        faithfully.
        """

        pos_R = initial_atoms.get_positions() / Bohr
        pos_P = final_atoms.get_positions() / Bohr
        r_R = LA.norm(pos_R[:, None, :] - pos_R[None, :, :], axis=-1)
        r_P = LA.norm(pos_P[:, None, :] - pos_P[None, :, :], axis=-1)

        # target distances, and weights set by the shorter of the two ends
        r_target = (1 - f) * r_R + f * r_P
        #r_short = np.minimum(r_R, r_P)
        r_short = r_R
        pairs = np.triu(np.ones_like(r_target, dtype=bool), k=1)
        fit = np.isin(initial_atoms.get_chemical_symbols(),
                      list(lst_elements))
        if fit.sum() > 1:
            pairs &= fit[:, None] & fit[None, :]
        bonded = pairs & (r_short < cutoff / Bohr)
        if bonded.any():
            pairs = bonded
        weights = np.where(pairs, 1 / np.where(pairs, r_short, 1) ** 4, 0)

        # start from, and stay close to, the structure we are stepping away
        # from rather than a point on the straight line to the product
        pos_ref = pos_R
        idx = self.indices

        def build(x):
            pos = pos_ref.copy()
            pos[idx, :] = x.reshape(-1, 3)
            return pos

        def S_and_grad(x):
            pos = build(x)
            diff = pos[:, None, :] - pos[None, :, :]
            r = LA.norm(diff, axis=-1)
            resid = np.where(pairs, r - r_target, 0)
            S = np.sum(weights * resid ** 2)
            # d/dx_a of the distance terms, summed over partners b
            coeff = 2 * weights * resid / np.where(r > 0, r, 1)
            coeff = coeff + coeff.T
            grad = np.einsum('ab,abk->ak', coeff, diff)
            # weak harmonic anchor to the reference structure
            dx = pos - pos_ref
            S += 1e-6 * np.sum(dx ** 2)
            grad += 2e-6 * dx
            return S, grad[idx, :].reshape(-1)

        result = minimize(S_and_grad, pos_ref[idx, :].reshape(-1),
                          jac=True, method='L-BFGS-B')
        disp = result.x.reshape(-1, 3) - pos_R[idx, :]
        bias_vector = self.normalize(disp.reshape(-1))
        return bias_vector

    def _get_bond_bias_vector(self, initial_atoms, final_atoms):
        """Bias direction built directly from the bonds that change.

        Every pair of atoms contributes the cartesian vector along that
        bond, scaled by how far the bond still has to go,
        r_ab(final) - r_ab(current): the two atoms are pushed apart if the
        bond lengthens and together if it shortens.  No pair is selected or
        excluded, by element, distance or anything else -- a bond already at
        its final length carries a change of zero and so contributes nothing
        on its own, which is why no cutoff or tolerance is needed.

        The contributions are simply summed, so an atom pulled by several
        changing bonds ends up moving along their resultant and nothing
        constrains the total displacement.  The overall scale is immaterial
        because the result is normalized.

        Unlike LST there is no fit, so nothing here tries to reach the final
        structure -- it only answers which way the changing bonds pull the
        atoms right now.
        """

        pos_R = initial_atoms.get_positions()
        pos_P = final_atoms.get_positions()
        r_R = LA.norm(pos_R[:, None, :] - pos_R[None, :, :], axis=-1)
        r_P = LA.norm(pos_P[:, None, :] - pos_P[None, :, :], axis=-1)

        pairs = np.triu(np.ones_like(r_R, dtype=bool), k=1)
        # amplitude of each bond's pull, shared evenly by its two atoms and
        # applied to both of them, so the sum runs over every pair at once
        amplitude = np.where(pairs, 0.5 * (r_P - r_R), 0)
        amplitude = amplitude + amplitude.T
        along = (pos_R[:, None, :] - pos_R[None, :, :]) / np.where(
            r_R > 0, r_R, 1)[:, :, None]
        disp = np.einsum('ab,abk->ak', amplitude, along)

        disp = disp[self.indices, :]
        if LA.norm(disp) < 1e-10:
            raise ValueError('every bond is already at its final length; '
                             'the bias direction is undefined')
        bias_vector = self.normalize(disp.reshape(-1))
        return bias_vector

    def _get_bias_vector(self, initial_atoms: Atoms,
                         final_atoms: Atoms) -> np.ndarray:
        """Bias direction from whichever interpolation was requested."""
        if self.interp_method == 'idpp':
            return self._get_idpp_bias_vector(initial_atoms, final_atoms)
        if self.interp_method == 'lst':
            return self._get_lst_bias_vector(initial_atoms, final_atoms)
        if self.interp_method == 'bond':
            return self._get_bond_bias_vector(initial_atoms, final_atoms)
        raise ValueError(f"unknown interp_method {self.interp_method!r}, "
                         "expected 'idpp', 'lst' or 'bond'")

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
