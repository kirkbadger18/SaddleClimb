import numpy as np
import numpy.linalg as LA
from ase.io.trajectory import Trajectory
class SaddleClimb:

    def __init__(self, atoms_initial, atoms_final, indices, calculator, hessian, fmax=0.01):

        self.atoms_initial = atoms_initial
        self.atoms_final = atoms_final
        self.indices = indices
        self.hessian = hessian
        self.calculator = calculator
        self.fmax=fmax

    def hessian_update(self, B_old, dF, dx):
        eig, vec = LA.eigh(B_old)
        dxT = np.transpose(dx)
        dFT = np.transpose(dF)
        B_abs = np.zeros(np.shape(self.hessian))
        for i in range(len(eig)):
            B_abs += eig[i] * np.outer(vec[i], np.transpose(vec[i]))
        dx_square = np.outer(dx, dxT)
        dF_square = np.outer(dF, dFT)
        M = dF_square + np.matmul(B_abs, np.matmul(dx_square, B_abs))
        j = dF - np.matmul(B_old, dx)
        u_term = 1/(np.matmul(dxT,np.matmul(M,dx)))
        u = np.matmul(M, dx * u_term)
        E_a = np.outer(u, np.transpose(j))
        print(E_a)
        E_b = np.outer(j, np.transpose(u))
        E_c = np.matmul(E_a, np.outer(dx, np.transpose(u)))
        E = E_a + E_b + E_c
        B = B_old + E
        return B

    def get_step(self, B, F, DX):
        eig, vec = LA.eigh(B) 
        eig = np.abs(eig)
        s_mag = np.matmul(vec,-1*F)
        s=np.zeros(np.shape(vec))
        for i in range(len(eig)):
            s[:,i]=s_mag[i]*vec[:,i]
        s_dot_DX=np.matmul(s, DX)

        for i in range(len(eig)):
            if s_dot_DX[i] > 1:
                eig[i] *= -1
                break
        H_step = np.matmul(vec,np.matmul(np.diag(eig),LA.inv(vec)))
        step = np.matmul(LA.inv(H_step), F)
        step_2D = np.zeros([len(self.indices), 3])
        for i in range(len(self.indices)):
            step_2D[i,:] = step[3*i:3*i+3]
        return step_2D

    def run(self):

        # first step
        atoms = self.atoms_initial.copy()
        F0 = self.atoms_initial.get_forces()
        DX = (self.atoms_final.positions - atoms.positions)[self.indices, :]
        dx = (1/LA.norm(DX))*DX
        atoms.positions[self.indices, :] += dx
        atoms.calc = self.calculator
        F = atoms.get_forces()
        dF = (F-F0)[self.indices, :]
        DX -= dx 
        B = self.hessian_update(self.hessian, dF.reshape(-1), dx.reshape(-1))

        # next steps
        traj = Trajectory('SaddleClimb.traj','w')
        traj.write(atoms)
        dFmax = 1
        while dFmax > self.fmax:
            dx = self.get_step(B, F[self.indices, :].reshape(-1), DX.reshape(-1))
            if np.max(np.abs(dx)) > 0.2:
                scale = 0.2/np.max(np.abs(dx))
                dx *=scale
            atoms.positions[self.indices, :] += dx
            F0 = F
            F = atoms.get_forces()
            traj.write(atoms)
            dF = (F-F0)[self.indices, :]
            dFmax = np.max(np.abs(dF))
            DX += dx
            B = self.hessian_update(B, dF.reshape(-1), dx.reshape(-1))



