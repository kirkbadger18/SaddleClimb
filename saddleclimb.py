import sys
import numpy as np
import numpy.linalg as LA
from numpy import matmul as mult
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

class SaddleClimb:

    def __init__(self, atoms_initial, atoms_final, calculator, indices, fmax=0.01):

        self.atoms_initial = atoms_initial
        self.atoms_final = atoms_final
        self.indices = indices
        self.hessian = 70 * np.eye(3*len(self.indices))
        self.calculator = calculator
        self.fmax=fmax
        self.forward_climb = True

    def TS_BFGS(self, B_old, dF, dx):
        eig, vec = LA.eigh(B_old)
        dxT = np.transpose(dx)
        dFT = np.transpose(dF)
        B_abs = np.zeros(np.shape(self.hessian))
        for i in range(len(eig)):
            B_abs += np.abs(eig[i]) * np.outer(vec[:,i], np.transpose(vec[:,i]))
        dx_square = np.outer(dx, dxT)
        dF_square = np.outer(dF, dFT)
        M = dF_square + np.matmul(B_abs, np.matmul(dx_square, B_abs))
        j = dF - np.matmul(B_old, dx)
        u_term = 1/(np.matmul(dxT,np.matmul(M,dx)))
        u = u_term * np.matmul(M, dx)
        E_a = np.outer(u, np.transpose(j))
        E_b = np.outer(j, np.transpose(u))
        E_c = np.matmul(E_a, np.outer(dx, np.transpose(u)))
        E = E_a + E_b - E_c
        B = B_old + E
        return B

#    def TS_BFGS(self, B_old, dF, dx):
#        eig, vec = LA.eigh(B_old)
#        B_abs = mult(vec,mult(np.diag(np.abs(eig)),vec.T))
#        denomterm1 = mult(dF.T,dx)
#        denomterm2 = mult(dx.T,mult(B_abs,dx))
#        denom = denomterm1**2 + denomterm2**2
#        num1term1 = dF - mult(B_old,dx)
#        num1term2 = np.transpose((denomterm1 * dF + denomterm2 * mult(B_abs,dx)))
#        num1 = np.outer(num1term1,num1term2)
#        num2 = np.outer(num1term2.T,num1term1.T)
#        num3term1 = denomterm1-mult(dx.T,mult(B_old,dx))
#        num3term2 = np.copy(num1term2.T)
#        num3term3 = np.copy(num1term2)
#        num3 = num3term1 * np.outer(num3term2,num3term3)
#        correction = (1/denom) * (num1 + num2 - num3)
#        B = B_old + correction
#        return B


    def ellipse_tangent_path(self,pos_1D):
        """
        ellipse equation:
        (x/a)**2 + (y/b)**2 = 1
        and its tangent:
        dy/dx = -(x*b**2)/(y*a**2)
        """
        pos = pos_1D
        idx = self.indices
        pos_i = self.atoms_initial.positions[idx,:].reshape(-1)
        pos_f = self.atoms_final.positions[idx,:].reshape(-1)
        y_vec = normalize(pos_f-pos_i)
        x_vec = normalize(pos - np.dot(pos,y_vec) * y_vec)
        center = pos_i + 0.5 * (pos_f - pos_i)
        b = LA.norm(0.5*(pos_f-pos_i))
        p1 = [0,b]
        p2 = [0,-b]
        p3 = [np.dot((pos-center),x_vec), np.dot((pos-center),y_vec)]
        if np.abs(p3[0]) >= 1e-4 and np.abs(p3[1]) < b:     
            a = np.abs(p3[0])/np.sqrt(1-(p3[1]/b)**2)
            dydx = (-p3[0]*b**2)/(p3[1]*a**2)
            path = normalize(dydx * y_vec + x_vec)
        else:
            path = normalize(pos_f-pos_i)
        if np.dot(path,(pos_f-pos)) < 0:
            path *= -1
        return path

    def ascent_step(self, B, F, path):
        eig, vec = LA.eigh(B)
        print('eig: ', eig)
        eig = np.abs(eig)
        F_norm = normalize(F)
        uphill_vec=np.zeros(np.shape(vec))
        for i in range(len(eig)):
            sign = np.sign(np.dot(vec[:,i],-1*F_norm))
            uphill_vec[:,i] = sign * vec[:,i]

        uphill_vec_path_dot=np.matmul(uphill_vec.T, path)
#        if self.forward_climb:
#            max_val = np.max(uphill_vec_path_dot)
#            eig_idx = np.where(uphill_vec_path_dot == max_val)
#        else:
#            min_val = np.min(uphill_vec_path_dot)
#            eig_idx = np.where(uphill_vec_path_dot == min_val)
#        eig[eig_idx] *= -1
#        print('eig_idx: ', eig_idx)

        max_diff = 0 #np.sqrt(1/len(eig))
        #found_uphill=False
        for i in range(len(eig)):
            if self.forward_climb and uphill_vec_path_dot[i] > max_diff:
                eig[i] *= -1
        #        found_uphill = True
                break
            elif not self.forward_climb and uphill_vec_path_dot[i] < -max_diff:
                eig[i] *= -1
       #         found_uphill = True
                break
        eig[i] = scale_climb(eig[i], vec[i], F)
        #print('Found uphill?: ',found_uphill)
        H_step = np.matmul(vec,np.matmul(np.diag(eig),LA.inv(vec)))
        step = -1 * np.matmul(LA.inv(H_step), -1*F)
        return step

    def descent_step(self, B, F):
        eig, vec = LA.eigh(B)
        eig = np.abs(eig)
        H_step = np.matmul(vec,np.matmul(np.diag(eig),LA.inv(vec)))
        step = np.matmul(LA.inv(H_step), F)
        return step

    def take_step(self,atoms,dx):
        if np.max(np.abs(dx)) > 0.1:
            scale = 0.1/np.max(np.abs(dx))
            dx *= scale
        idx = self.indices
        atoms.positions[idx,:] += dx
        return atoms, dx

    def test_run(self):
        traj = Trajectory('SaddleClimb.traj','w')
        atoms = self.atoms_initial.copy()
        constraints = self.atoms_initial.constraints.copy()
        atoms.set_constraint(constraints)
        atoms.calc = self.calculator
        idx = self.indices
        F_calc = atoms.get_forces()
        F = F_calc[idx,:]
        traj.write(atoms)
        B = self.hessian
        pos_f_1D = self.atoms_final.positions[idx,:].reshape(-1)
        pos_i_1D = self.atoms_initial.positions[idx,:].reshape(-1)
        dx_1D = 0.01 * normalize(pos_f_1D-pos_i_1D)
        dx = dx_1D.reshape(-1,3)
        Fmax = 1
        dxi = 0
        n = 0
        while Fmax > self.fmax or dxi < 0.5:
            n += 1
            atoms, dx = self.take_step(atoms,dx)
            pos_1D = atoms.positions[idx,:].reshape(-1)
            path = self.ellipse_tangent_path(pos_1D)
            write_path(atoms,idx, path, 'path_{}.traj'.format(str(n)))
            dxi = LA.norm(pos_i_1D-pos_1D)
            print('distance to initial position:\n', dxi)
            F0 = F
            F_calc = atoms.get_forces()
            F = F_calc[idx,:]
            dF = (F-F0)
            Fmax = np.max(np.abs(F))
            print('Fmax\n', Fmax)
            traj.write(atoms)

            rev_dot = np.dot(F.reshape(-1), -1*path)
            if Fmax < self.fmax and dxi < 0.1:
                pass
            elif rev_dot > 0:
                self.forward_climb = True
            else:
                self.forward_climb = False
            print('self.forward_climb?: ', self.forward_climb)
            B = self.TS_BFGS(B, -1*dF.reshape(-1), dx.reshape(-1))
            path_F_dot = np.abs(np.dot(path, normalize(F.reshape(-1))))
            print('path_F_dot: ',path_F_dot)
#            if path_F_dot < 0.1:
#                F_proj = project_out(F.reshape(-1),path)
#                dx_1D = self.descent_step(B, F_proj)
#            else:
            dx_1D = self.ascent_step(B, F.reshape(-1), path)
            dx = dx_1D.reshape(-1,3)
            sys.stdout.flush()

def scale_climb(eig, vec, F):
    g = np.dot(vec, -1*F)
    step_len = np.abs((1/eig) * LA.norm(g))
    if step_len > 0.1:
        scale = 0.1 / step_len
        print('scale of eig: ',1/scale)
        eig *= 1/scale
    return eig

def normalize(v):
    norm = LA.norm(v)
    return v / norm

def project_out(vec,direction):
    mag = np.dot(vec,direction)
    new_vec = vec - mag * direction
    return vec

def write_path(atoms,idx,path,name):
    path_traj = Trajectory(name,'w')
    points = 0.6 * np.sin(np.linspace(0,2*np.pi,30))    
    for point in points:
        atoms_cpy = atoms.copy()
        disp = point * path
        atoms_cpy.positions[idx,:] += disp.reshape(-1,3)
        path_traj.write(atoms_cpy)
