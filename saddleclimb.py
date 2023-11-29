import numpy as np
import numpy.linalg as LA
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

class SaddleClimb:

    def __init__(self, atoms_initial, atoms_final, indices, calculator, hessian, fmax=0.01):

        self.atoms_initial = atoms_initial
        self.atoms_final = atoms_final
        self.indices = indices
        self.hessian = hessian
        self.calculator = calculator
        self.fmax=fmax
        self.forward_climb = True

    def TS_BFGS(self, B_old, dF, dx):
        eig, vec = LA.eigh(B_old)
        dxT = np.transpose(dx)
        dFT = np.transpose(dF)
        B_abs = np.zeros(np.shape(self.hessian))
        for i in range(len(eig)):
            B_abs += np.abs(eig[i]) * np.outer(vec[i], np.transpose(vec[i]))
        dx_square = np.outer(dx, dxT)
        dF_square = np.outer(dF, dFT)
        M = dF_square + np.matmul(B_abs, np.matmul(dx_square, B_abs))
        j = dF - np.matmul(B_old, dx)
        u_term = 1/(np.matmul(dxT,np.matmul(M,dx)))
        u = np.matmul(M, dx * u_term)
        E_a = np.outer(u, np.transpose(j))
        E_b = np.outer(j, np.transpose(u))
        E_c = np.matmul(E_a, np.outer(dx, np.transpose(u)))
        E = E_a + E_b - E_c
        B = B_old + E
        return B

    def ascent_step(self, B, F, DX):
        eig, vec = LA.eigh(B)
        eig = np.abs(eig)
        F_norm = (1/LA.norm(F)) * F
        DX_norm = (1/LA.norm(DX)) * DX        
        s_mag = np.matmul(vec,-1*F_norm)
        s=np.zeros(np.shape(vec))
        for i in range(len(eig)):
            s[:,i]=np.sign(s_mag[i])*vec[:,i]
        s_dot_DX=np.matmul(np.transpose(s), DX_norm)
        max_diff = np.sqrt(1/len(eig))
        for i in range(len(eig)):
            if self.forward_climb and s_dot_DX[i] > max_diff:
                eig[i] *= -1
                break
            elif not self.forward_climb and s_dot_DX[i] < -max_diff:
                eig[i] *= -1
                break
        H_step = np.matmul(vec,np.matmul(np.diag(eig),LA.inv(vec)))
        step = np.matmul(LA.inv(H_step), F)
        step_2D = np.zeros([len(self.indices), 3])
        for i in range(len(self.indices)):
            step_2D[i,:] = step[3*i:3*i+3]
        return step_2D

    def descent_step(self, B, F):
        eig, vec = LA.eigh(B)
        eig = np.abs(eig)
        H_step = np.matmul(vec,np.matmul(np.diag(eig),LA.inv(vec)))
        step = np.matmul(LA.inv(H_step), F)
        step_2D = np.zeros([len(self.indices), 3])
        for i in range(len(self.indices)):
            step_2D[i,:] = step[3*i:3*i+3]
        return step_2D

    def run(self):
        traj = Trajectory('SaddleClimb.traj','w')
        sp = Trajectory('Stationary_points.traj','w')
        atoms = self.atoms_initial.copy()
        atoms.calc = self.calculator
        F = atoms.get_forces()
        traj.write(atoms)
        sp.write(atoms)
        DXf = (self.atoms_final.positions - atoms.positions)[self.indices, :]
        n = 0
        B = self.hessian
        while LA.norm(DXf) > 0.1:
            self.forward_climb = True
            n += 1
            DXi = np.zeros(np.shape(DXf))
            dx = (0.01 / LA.norm(DXf)) * DXf
            Fmax = 1
            while Fmax > self.fmax or LA.norm(DXi) < 0.1:
                if np.max(np.abs(dx)) > 0.1:
                    scale = 0.1/np.max(np.abs(dx))
                    dx *= scale
                atoms.positions[self.indices, :] += dx
                F0 = F
                F = atoms.get_forces()
                dF = (F-F0)[self.indices, :]
                Fmax = np.max(np.abs(F[self.indices, :]))
                traj.write(atoms)
                DXf -= dx
                DXi -= dx
                fwrd_dot = np.dot(np.transpose(F[self.indices, :].reshape(-1)), DXf.reshape(-1))
                rev_dot = np.dot(np.transpose(F[self.indices, :].reshape(-1)), DXi.reshape(-1)) 
                if rev_dot < 0:
                    self.forward_climb = False
                    DX = -DXi
                elif rev_dot > 0:
                    self.forward_climb = True
                    DX = DXf
                B = self.TS_BFGS(B, dF.reshape(-1), dx.reshape(-1))
                dx = self.ascent_step(B, F[self.indices, :].reshape(-1), DX.reshape(-1))
            sp.write(atoms)

            dx = (-0.01 / LA.norm(DXi)) * DXi
            DXi = np.zeros(np.shape(DXf))
            Fmax = 1 
            while Fmax > self.fmax or LA.norm(DXi) < 0.1:
                DXi -= dx
                DXf -= dx
                if np.max(np.abs(dx)) > 0.1:
                    scale = 0.1/np.max(np.abs(dx))
                    dx *=scale
                atoms.positions[self.indices, :] += dx
                F0 = F
                F = atoms.get_forces()
                dF = (F-F0)[self.indices, :]
                Fmax = np.max(np.abs(F[self.indices, :]))
                traj.write(atoms)
                B = self.TS_BFGS(B, dF.reshape(-1), dx.reshape(-1))
                dx = self.descent_step(B, F[self.indices, :].reshape(-1))
            sp.write(atoms)

  
