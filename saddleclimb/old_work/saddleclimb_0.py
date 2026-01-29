import sys
import os
import scipy
import numpy as np
import numpy.linalg as LA
from numpy import matmul as mult
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.io.trajectory import Trajectory
from pathlib import Path


class SaddleClimb:

    def __init__(
            self: None,
            atoms_initial: Atoms,
            atoms_final: Atoms,
            calculator: Calculator,
            indices: list,
            fmax: float = 0.01,
            delta0: float = 1e-2,
            rho_dec: float = 5,
            rho_inc: float = 1.035,
            sigma_inc: float = 1.15,
            sigma_dec: float = 0.65,
            eta: float = 1e-3,
            alpha0: float = 1,
            logfile: str = 'climb.log',
            trajfile: str = 'climb.traj',
            ) -> None:

        self.atoms_initial = atoms_initial
        self.atoms_final = atoms_final
        self.indices = indices
        self.hessian = 100 * np.eye(3*len(self.indices))
        self.calculator = calculator
        self.fmax = fmax
        self.delta = delta0
        self.rho_inc = rho_inc
        self.rho_dec = rho_dec
        self.sigma_inc = sigma_inc
        self.sigma_dec = sigma_dec
        self.eta = eta
        self.alpha = alpha0
        self.logfile = logfile
        self.trajfile = trajfile
        self.forward_climb = True

    def update_hessian(
            self: None, B_old: np.ndarray,
            dg: np.ndarray, dx: float
            ) -> np.ndarray:
        """
        Hessian update procedure described by:

        """
        eig, vec = LA.eigh(B_old)
        print(eig)
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

    def get_ellipse_tangent(self: None, pos_1D: np.ndarray, g) -> np.ndarray:
        """
        ellipse equation:
        (x/a)**2 + (y/b)**2 = 1
        and its tangent:
        dy/dx = -(x*b**2)/(y*a**2)
        """
        pos = pos_1D
        idx = self.indices
        pos_i = self.atoms_initial.positions[idx, :].reshape(-1)
        pos_f = self.atoms_final.positions[idx, :].reshape(-1)
        y_vec = normalize(pos_f-pos_i)
        x_vec = normalize(pos - np.dot(pos, y_vec) * y_vec)
        center = pos_i + 0.5 * (pos_f - pos_i)
        p1 = np.array([0, np.dot((pos_i-center), y_vec)])
        p2 = np.array([0, np.dot((pos_f-center), y_vec)])
        p3 = np.array([np.dot((pos-center), x_vec), np.dot((pos-center), y_vec)])
        l1 = LA.norm(p3 - p1)
        l2 = LA.norm(p3 - p2)
        print('p1, p2, p3: ',p1, p2, p3)
        vert1 = np.array([0,p2[1] - (l1 + l2)]) 
        vert2 = np.array([0,p1[1] + (l1 + l2)])
        b = 0.5 * (vert2-vert1)[1]
               #if np.abs(p3[0]) >= 1e-4 and np.abs(p3[1]) < b:
        if np.abs(p3[1] - b) > 1e-4:
            a= np.abs(p3[0]) / np.sqrt(1 - (p3[1] / b) ** 2)
            dydx = (-p3[0] * b ** 2) / (p3[1] * a ** 2)
            path = normalize(dydx * y_vec + x_vec)
            print('p3, a, b', p3, a, b)
            print('dydx: ',dydx)
        else:
            path = normalize(pos_f-pos_i)
        for i in range(int(len(pos)/3)):
            atom_path = path[3*i:3*i+3]
            atom_g = g[3*i:3*i+3]
            if np.dot(atom_path, atom_g) < 0:
                path[3*i:3*i+3] *= -1
        return path

    def partition_hessian(self: None, B: np.ndarray,
                        g: np.ndarray
                        ) -> np.ndarray:

        eig, vec = LA.eigh(B)
        g_norm = normalize(g)
        uphill_vec = np.zeros(np.shape(vec))
        for i in range(len(eig)):
            sign = np.sign(np.dot(vec[:, i], g_norm))
            uphill_vec[:, i] = sign * vec[:, i]
        uphill_vec_path_dot = mult(uphill_vec.T, self.path)
        #mindot = 1/np.sqrt(len(eig))
        #for i in range(len(eig)):
            #if self.forward_climb and uphill_vec_path_dot[i] > mindot:
            #if uphill_vec_path_dot[i] > mindot:
        maxdot = np.max(uphill_vec_path_dot)
        eigidx = np.where(np.isclose(uphill_vec_path_dot,maxdot))
                #eigidx = i
                #break
            #elif not self.forward_climb and uphill_vec_path_dot[i] < -mindot:
                #mindot = np.min(uphill_vec_path_dot)
                #eigidx = np.where(np.isclose(uphill_vec_path_dot,mindot))
                #eigidx = i
                #break
        climb_vec = vec[:,eigidx[0][0]]
        descend_idx = [val for val in range(len(vec)) if val != eigidx[0][0]]
        descend_vec = vec[:,descend_idx]
        return climb_vec, descend_vec

    def get_s(self, vec_min, vec_max, g, B, a):
        """ PRFO from Sella paper"""
        vBv_min = mult(vec_min.T, mult(B, vec_min))
        vBv_max = mult(vec_max.T, mult(B, vec_max))
        M_min = np.zeros([len(vBv_min)+1, len(vBv_min)+1])
        M_max = np.zeros([2, 2])
        M_min[0:-1,0:-1]= a ** 2 * vBv_min
        M_min[-1,0:-1] = a * mult(g.T, vec_min)
        M_min[0:-1,-1] = a * mult(vec_min.T, g)
        M_max[0,0]= a ** 2 * vBv_max
        M_max[1,0] = a * mult(g.T, vec_max)
        M_max[0,1] = a * mult(vec_max.T, g)
        scale_min = 1 / LA.eigh(M_min)[1][-1,0]
        scale_max = 1 / LA.eigh(M_max)[1][1,1]
        s_max = a * scale_max * LA.eigh(M_max)[1][0,1]
        s_min = a * scale_min * LA.eigh(M_min)[1][0:-1,0]
        #print('last entries are 1??: ', scale_max, scale_min)
        sk = s_max * vec_max + mult(vec_min, s_min) 
        return sk

    def correct_step(self, step):
        if self.forward_climb and np.dot(step, self.path) < 0:
            step = project_out(step, self.path)
        elif not self.forward_climb and np.dot(step, self.path) > 0:
            step = project_out(step, self.path)
        return step


    def optimize_a(self, a, vec_min, vec_max, g, B):
        step = self.get_s(vec_min, vec_max, g, B, a)
        norm = LA.norm(step)
        length = np.sqrt(len(g)) * self.delta 
        diff = norm - length
        return diff

    def assess_trust_radius(self, dx, dE, g, B):
        pred = mult(g.T, dx) + 0.5 * mult(dx.T, mult(B, dx))
        rho = dE / pred
        rho_between = rho > 1 / self.rho_inc and rho < self.rho_inc
        rho_outside = rho < 1 / self.rho_dec or rho > self.rho_dec
        length = np.sqrt(len(g)) * self.delta
        l_min = np.sqrt(len(g)) * self.eta
        if rho_between and self.sigma_inc * LA.norm(dx) >= length:
            self.delta = self.sigma_inc * LA.norm(dx) / np.sqrt(len(g))
        elif rho_outside and self.sigma_dec * LA.norm(dx) > l_min:
            self.delta = self.sigma_dec * LA.norm(dx) / np.sqrt(len(g))
        elif rho_outside and self.sigma_dec * LA.norm(dx) < l_min:
            self.delta = self.eta
        print('rho is: ', rho)
        return

    def initialize_atoms(self: None) -> tuple[Atoms, np.ndarray, np.ndarray]:
        atoms = self.atoms_initial.copy()
        constraints = self.atoms_initial.constraints.copy()
        atoms.set_constraint(constraints)
        atoms.calc = self.calculator
        idx = self.indices
        B_init = self.hessian
        return atoms, idx, B_init

    def initialize_run(self: None, atoms: Atoms, idx: list) -> np.ndarray:
        traj = Trajectory(self.trajfile, 'w')
        g_init = atoms.get_forces()[idx, :].reshape(-1)
        E_init = atoms.calc.results['energy']
        traj.write(atoms)
        Fmax = np.max(np.abs(g_init))
        log_string = self.get_log_string(0, E_init, Fmax)
        self._log(log_string)
        return traj, g_init, E_init

    def get_initial_step(
            self: None, idx: list
            ) -> tuple[np.ndarray, np.ndarray]:

        pos_f_1D = self.atoms_final.positions[idx, :].reshape(-1)
        pos_i_1D = self.atoms_initial.positions[idx, :].reshape(-1)
        multiplier = np.sqrt(len(pos_i_1D) * self.delta ** 2)
        dx_1D = multiplier * normalize(pos_f_1D - pos_i_1D)
        dx_init = dx_1D.reshape(-1, 3)
        return dx_init, pos_i_1D

    def check_climb_direction(self: None, g: np.ndarray,
                              Fmax: float, dxi: float
                              ) -> None:
        idx = self.indices
        pos_f_1D = self.atoms_final.positions[idx, :].reshape(-1)
        pos_i_1D = self.atoms_initial.positions[idx, :].reshape(-1)
        dist = LA.norm(pos_f_1D - pos_i_1D)
        g_path_dot = np.dot(g, self.path)
        print('dxi, dist = ', dxi, dist)
        #if Fmax < self.fmax and dxi < 0.1:
        if dxi < 0.1 * dist:
            pass
        elif g_path_dot > 0:
            self.forward_climb = True
        else:
            self.forward_climb = False

    def get_log_string(self,n,E,Fmax):
        n_str = str(n).ljust(20)
        E_str = str(np.round(E,6)).ljust(20)
        F_str = str(np.round(Fmax,6)).ljust(20)
        #if self.forward_climb:
        #    climb = 'forward'
        #else:
        #    climb = 'reverse'
        log_string = n_str + E_str + F_str # + climb
        return log_string

    def _log(self: None, string: str) -> None:
        if self.logfile is None:
            print(string)
        else:
            with open(self.logfile,'a') as log:
                log.write(string + '\n')
        sys.stdout.flush()

    def initialize_logging(self: None):
        n_str = 'Iteration'.ljust(20)
        E_str = 'Energy (eV)'.ljust(20)
        F_str = 'Fmax (eV/A)'.ljust(20)
        climb = 'Climbing Direction'
        log_string = n_str + E_str + F_str + climb
        climb = Path(self.logfile)
        if climb.exists():
            os.remove(self.logfile)
        self._log(log_string)

    def run(self: None) -> None:
        atoms, idx, B = self.initialize_atoms()
        self.initialize_logging()
        traj, g, E = self.initialize_run(atoms, idx)
        dx, pos_i_1D = self.get_initial_step(idx)
        dx_1D = dx.reshape(-1)
        Fmax, dxi, n = 1, 0, 0
        while Fmax > self.fmax or dxi < 0.5:
            n += 1
            atoms.positions[idx, :] += dx
            pos_1D = atoms.positions[idx, :].reshape(-1)
            dxi = LA.norm(pos_i_1D - pos_1D)
            g0, E0 = g, E
            g = -atoms.get_forces()[idx, :].reshape(-1)
            E = atoms.calc.results['energy']
            dg, dE = (g - g0), (E - E0)
            Fmax = np.max(np.abs(g))
            self.assess_trust_radius(dx_1D, dE, g0, B)
            self.path = self.get_ellipse_tangent(pos_1D, g)
            self.write_path(atoms, idx, 'path_{}.traj'.format(str(n)))
            #self.check_climb_direction(g, Fmax, dxi)
            B = self.update_hessian(B, dg, dx_1D)
            vec_max, vec_min = self.partition_hessian(B, g)
            a = 1
            dx_1D = self.get_s(vec_min, vec_max, g, B, a)
            #dx_1D = self.correct_step(dx_1D)
            max_l = np.sqrt(len(dx_1D) * self.delta ** 2)

            if LA.norm(dx_1D) > max_l:
                args = vec_min, vec_max, g, B
                a_sol = scipy.optimize.root(self.optimize_a, a, args)
                a = a_sol.x
                dx_1D = self.get_s(vec_min, vec_max, g, B, a) 
            print('dx norm: ', LA.norm(dx_1D))
            print('alpha: ', a)
            print('delta: ', self.delta)
            #dx_1D = self.correct_step(dx_1D)
            #print('corrected dx norm: ', LA.norm(dx_1D))
            
            dx = dx_1D.reshape(-1, 3)
            log_string = self.get_log_string(n,E,Fmax)
            self._log(log_string)
            traj.write(atoms) 

    def write_path(self: None, atoms: Atoms, idx: list, name: str) -> None:
        path_traj = Trajectory(name, 'w')
        points = 0.6 * np.sin(np.linspace(0, 2*np.pi, 30))
        for point in points:
            atoms_cpy = atoms.copy()
            disp = point * self.path
            atoms_cpy.positions[idx, :] += disp.reshape(-1, 3)
            path_traj.write(atoms_cpy)

def normalize(v: np.ndarray) -> np.ndarray:
    norm = LA.norm(v)
    return v / norm

def project_out(vec: np.ndarray, direction: np.ndarray) -> np.ndarray:
    mag = np.dot(vec, direction)
    new_vec = vec - mag * direction
    return new_vec


