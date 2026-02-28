import sys
import os
import numpy as np
import numpy.linalg as LA
from numpy import matmul as mult
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.io.trajectory import Trajectory
from pathlib import Path
import copy


class SaddleClimb:

    def __init__(
            self: None,
            atoms_initial: Atoms,
            atoms_final: Atoms,
            calculator: Calculator,
            indices: list = None,
            fmax: float = 0.01,
            maxstepsize: float = 0.2,
            delta0: float = 0.05,
            logfile: str = 'climb.log',
            trajfile: str = 'climb.traj',
            ) -> None:

        self.atoms_initial = atoms_initial
        self.atoms_final = atoms_final
        self.indices = indices
        self.calculator = calculator
        self.fmax = fmax
        self.maxstepsize = maxstepsize
        self.delta = delta0
        self.logfile = logfile
        self.trajfile = trajfile
        self._restart = False
        if not self.indices:
            self._get_moving_atoms()
        self.hessian = 100 * np.eye(3*len(self.indices))

    def _get_moving_atoms(self):
        dpos = self.atoms_final.positions - self.atoms_initial.positions
        idx = []
        for i in range(dpos.shape[0]):
            if LA.norm(dpos[i, :]) > 1e-6:
                idx.append(i)
        self.indices = idx.copy()

    def _get_step(self, B, g, pos_1D):
        dxi = self._pos_i_1D - pos_1D
        dxf = self._pos_f_1D - pos_1D
        dxi_to_f = self._pos_f_1D - self._pos_i_1D
        eigs_B, vecs_B = LA.eigh(B)
        self._climbing = True
        if np.dot(g, dxi) < 0 and np.dot(g, dxf) < 0:
            eigs_tmp, vecs_tmp = eigs_B.copy(), vecs_B.copy()
            self._climbing = False
        elif eigs_B[0] < 0:
            eigs_tmp, vecs_tmp = eigs_B.copy(), vecs_B.copy()
        else:
            vecs_tmp, R = LA.qr(dxi_to_f.reshape(len(dxi_to_f), 1),
                                mode='complete')
            B_transformed = mult(vecs_tmp.T, mult(B, vecs_tmp))
            eigs_tmp = np.diag(B_transformed).copy()
        for i, eig in enumerate(eigs_tmp):
            if i == 0 and self._climbing:
                eigs_tmp[i] = - np.abs(eig)
            else:
                eigs_tmp[i] = np.abs(eig)
        Dmat = np.diag(eigs_tmp)
        B_temp = mult(vecs_tmp, mult(Dmat, vecs_tmp.T))
        sys.stdout.flush()
        inv_B_temp = LA.inv(B_temp)
        dx_1D = -mult(inv_B_temp, g)
        maxstep = 0
        for i in range(len(self.indices)):
            stepsize = LA.norm(dx_1D[3*i:3*i+3])
            if stepsize > maxstep:
                maxstep = stepsize
        if maxstep > self.maxstepsize:
            dx_1D *= self.maxstepsize / maxstep

        return dx_1D

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
        #B_init = B_init.reshape(atoms.info["saddleclimb_hessian_shape"])
        #B_init = self._restart_trajectory.info['saddleclimb_hessian'].copy()
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
        Fmax = np.max(np.abs(g))
        return traj, g, E, Fmax

    def _get_initial_step(
            self: None, idx: list
            ) -> tuple[np.ndarray, np.ndarray]:
        self._pos_f_1D = self.atoms_final.positions[idx, :].reshape(-1)
        self._pos_i_1D = self.atoms_initial.positions[idx, :].reshape(-1)
        dx_1D = self.delta * self.normalize(self._pos_f_1D - self._pos_i_1D)
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
            atoms, idx, B = self._initialize_atoms_restart()
            traj, g, E, Fmax = self._initialize_run_restart(idx)
            self._pos_f_1D = self.atoms_final.positions[idx, :].reshape(-1)
            self._pos_i_1D = self.atoms_initial.positions[idx, :].reshape(-1)
            dx_1D = self._get_step(B, g, atoms.positions[idx, :].reshape(-1))
            dx = dx_1D.reshape(-1, 3)
            pos_1D = atoms.positions[idx, :].reshape(-1)
            dxi = LA.norm(self._pos_i_1D - pos_1D)
            n = self._restart_trajectory.info['saddleclimb_iterations']
        else:
            atoms, idx, B = self._initialize_atoms()
            traj, g, E = self._initialize_run(atoms, idx)
            dx, dx_1D = self._get_initial_step(idx)
            Fmax, dxi, n = 1, 0, 0
        while Fmax > self.fmax or dxi < 0.5:
            atoms.positions[idx, :] += dx
            pos_1D = atoms.positions[idx, :].reshape(-1)
            dxi = LA.norm(self._pos_i_1D - pos_1D)
            g0 = g
            g = -self._get_F(atoms)[idx, :].reshape(-1)
            E = atoms.calc.results['energy']
            dg = g - g0
            Fmax = np.max(np.abs(g))
            B = self._update_hessian(B, dg, dx_1D)
            dx_1D = self._get_step(B, g, pos_1D)
            dx = dx_1D.reshape(-1, 3)
            n += 1
            log_string = self._get_log_string(n, E, Fmax)
            self._log(log_string)
            atoms.info["saddleclimb_hessian"] = B.tolist()
            atoms.info["saddleclimb_hessian_shape"] = B.shape
            #atoms.info['saddleclimb_hessian'] = B.copy()
            atoms.info['saddleclimb_iterations'] = n + 0
            traj.write(atoms)
            if maxsteps and n >= maxsteps:
            #    self._log('maxsteps reached, terminating!')
                break
            #if Fmax < self.fmax and dxi > 0.5:
            #    self._log('Optimization complete!')

    def restart_climb(self, restart_trajectory: Atoms):
        assert 'saddleclimb_hessian' in restart_trajectory.info
        self._restart = True
        self._restart_trajectory = copy.deepcopy(restart_trajectory)
        self.climb()

    def normalize(self: None, v: np.ndarray) -> np.ndarray:
        norm = LA.norm(v)
        return v / norm
