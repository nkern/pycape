"""
finite_diff.py

Finite difference approximations to gradients

https://en.wikipedia.org/wiki/Finite_difference

Nicholas Kern
"""
import numpy as np
import scipy.linalg as la
import traceback

__all__ = ['FiniteDiff']

class FiniteDiff(object):

    def __init__(self):
        pass

    def first_central(self, f_neg, f_pos, dx):
        """
        Central finite difference for first order partial derivative of f

        f_neg : scalar
                f(x-dx,y,z,..)

        f_pos : scalar
                f(x+dx,y,z,..)

        dx    : scalar
                dx
        """
        return (f_pos - f_neg)/(2*dx)

    def first_forward(self, f, f_pos, dx):
        """
        Forward finite difference for first order partial derivative of f

        f     : scalar
                f(x,y,z,..)

        f_pos : scalar
                f(x+dx,y,z,..)

        dx    : scalar
                dx
        """
        return (f_pos-f)/dx


    def second_central(self, f, f_neg1, f_pos1, f_neg2, f_pos2, f_neg1_neg2, f_pos1_pos2, dx1, dx2):
        """
        Central finite difference approximation for second order partial derivative of f

        f           : scalar
                    f(x,y,z,..)

        f_neg1      : scalar
                    f(x-dx1,y,z,..)

        f_pos1      : scalar
                    f(x+dx1,y,z,..)

        f_neg2      : scalar
                    f(x,y-dx2,z,..)

        f_pos2      : scalar
                    f(x,y+dx2,z,..)

        f_neg1_neg2 : scalar
                    f(x-dx1,y-dx2,z,..)

        f_pos1_pos2 : scalar
                    f(x+dx1,y+dx2,z,..)

        dx1         : scalar
                    dx1

        dx2         : scalar
                    dx2
        """
        return (f_pos1_pos2 - f_pos1 - f_pos2 + 2*f - f_neg1 - f_neg2 + f_neg1_neg2) / (2*dx1*dx2)

    def calc_jacobian(self, f, pos_vec, diff_vec, neg_vec=None):
        """
        Calculate the approximate Jacobian Matrix
        f           : scalar
                    f(x,y,z,..)

        pos_vec     : ndarray [dtype=float, shape=(ndim,)]
                    vector containing func positive offsets

        diff_vec    : ndarray [dtype=float, shape=(ndim,)]
                    vector containing dx offsets        

        neg_vec     : ndarray [dtype=float, shape=(ndim,), default=None]
                    vector containing func negative offsets
        """
        ndim = len(diff_vec)
        J = np.empty((1,ndim))
        for i in range(ndim):
            if neg_vec is None:
                J[0,i] = self.first_forward(f, pos_vec[i], diff_vec[i])
            else:
                J[0,i] = self.first_central(neg_vec[i], pos_vec[i], diff_vec[i])
        return J

    def calc_hessian(self, f, pos_mat, neg_mat, diff_vec, out_jacobian=True):
        """
        Calculate the approximate Hessian Matrix

        f           : scalar
            evaluation of "f" at fiducial point

        theta       : ndarray [dtype=float, shape=(ndim,)]
            Fiducial point in parameter space

        pos_mat     : ndarray [dtype=float, shape=(ndim,ndim)]
            A matrix holding evaluations of "f" at x1+dx1, x2+dx2, ...
            Diagonal is f_ii = f(..,xi+dxi,..). Example: f_11 = f(x1+dx1, x2, x3, ..)
            Off-diagonal is f_ij = f(..,xi+dxi,..,xj+dxj,..). Example: f_12 = f(x1+dx1, x2+dx2, x3, ..)

        neg_mat     : ndarray [dtype=float, shape=(ndim,ndim)]
            A matrix holding evaluations of "f" at x1-dx1, x2-dx2, ...
            Same format as pos_mat

        diff_vec    : ndarray [dtype=float, shape=(ndim,)]
            A vector holding the step size for each dimension. Example: (dx1, dx2, dx3, ...) 

        out_jacobian    : bool [default=True]
            If True: output jacobian matrix as well as hessian matrix
        """
        ndim = len(diff_vec)

        # Calculate Hessian Matrix via Finite Difference
        H = np.empty((ndim,ndim))
        if out_jacobian == True: J = np.empty((1,ndim))
        for i in range(ndim):
            for j in range(i, ndim):
                hess = self.second_central(f, neg_mat[i,i], pos_mat[i,i], neg_mat[j,j], pos_mat[j,j], neg_mat[i,j], pos_mat[i,j], diff_vec[i], diff_vec[j])
                H[i,j] = hess
                if i != j: H[j,i] = hess
                if out_jacobian == True and j==i: J[0,i] = self.first_central(neg_mat[i,i], pos_mat[i,i], diff_vec[i])

        if out_jacobian == True:
            return H, J
        else:
            return H

    def calc_partials(self, f, theta, diff_vec, second_order=True):
        """
        Use finite difference to calculate pos_mat and neg_mat
        """
        ndim = len(diff_vec)

        # Calculate positive and negative matrices
        pos_mat = np.empty((ndim,ndim))
        neg_mat = np.empty((ndim,ndim))
        for i in range(ndim):
            if second_order == True: second = ndim
            else: second = i+1
            for j in range(i,second):
                theta_pos   = theta + np.eye(ndim)[i] * diff_vec
                theta_neg   = theta - np.eye(ndim)[i] * diff_vec
                if j != i:
                    theta_pos   += np.eye(ndim)[j] * diff_vec
                    theta_neg   -= np.eye(ndim)[j] * diff_vec
                f_pos       = f(theta_pos)
                f_neg       = f(theta_neg)
                pos_mat[i,j] = 1 * f_pos
                neg_mat[i,j] = 1 * f_neg

        if second_order == True:
            return pos_mat, neg_mat
        else:
            return pos_mat.diagonal(), neg_mat.diagonal()

    def propose_O2(self, H,J,gamma=0.5):
        """
        Give a second order Newton-Raphson proposal step from current position theta given Hessian H and Jacobian J
        In order to find local minima
        """
        # Evaluate proposal step
        prop = -np.dot(la.inv(H),J.T).ravel()

        return gamma * prop

    def propose_O1(self, J, gamma=0.5):
        """
        Give a first order proposal step to minimize a function
        """
        prop = -gamma*J
        return prop

    def find_root(self, f, theta, diff_vec, nsteps=10, gamma=0.1, second_order=True):
        """
        Find root
        """
        steps = []
        for i in range(nsteps):
            try:
                steps.append(np.copy(theta))
                pos_mat, neg_mat = self.calc_partials(f, theta, diff_vec, second_order=second_order)
                if second_order == True:
                    H, J = self.calc_hessian(f(theta), pos_mat, neg_mat, diff_vec)
                else:
                    J = self.calc_jacobian(f(theta), pos_mat, neg_mat, diff_vec)
                if second_order == True:
                    prop = self.propose_O2(H, J[0], gamma=gamma)
                else:
                    prop = self.propose_O1(J[0], gamma=gamma)
                theta = 1*theta + prop
            except:
                print("Optimization failed... releasing steps already done")
                traceback.print_exc()
                return np.array(steps)

        return np.array(steps)

    def hess(self, theta, f, diff_vec):
        pos_mat, neg_mat = self.calc_partials(f, theta, diff_vec, second_order=True)
        H, J = self.calc_hessian(f(theta), pos_mat, neg_mat, diff_vec)
        return H

    def jac(self, theta, f, diff_vec):
        pos_vec, neg_vec = self.calc_partials(f, theta, diff_vec, second_order=False)
        J = self.calc_jacobian(f, pos_vec, diff_vec, neg_vec=neg_vec)
        return J[0]

