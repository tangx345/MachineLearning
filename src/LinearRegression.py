# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:48:35 2021

@author: Yang
"""

import numpy as np
import sys
sys.path.append('./MathUtils.py')
import MathUtils

class LS_Matrix:
    # This is the very basic inverse matrix least square linear regression fit
    # It requites the matrix to be non-sigular
    def __init__(self, X, Y, normalize = False):
        # input X must be n * m numpy array, even m = 1, it should be n * 1 ndarray
        # input Y must be n * 1 numpy array
        if len(X.shape) != 2:
            raise ValueError("LS_Matrix init failed, input X must be numpy ndarray!")
        if len(Y.shape) != 1:
            raise ValueError("LS_Matrix init failed, input Y must be numpy 1d array!")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("LS_Matrix init failed, input X and Y must have the same number of rows!")
        
        N = X.shape[0]
        m = X.shape[1]
        if normalize:
            self.mu = np.mean(X, axis = 0)
            self.std = np.std(X, axis = 0)
            self.X = (X - self.mu) / self.std
        else:
            self.mu = np.zeros(m)
            self.std = np.ones(m) 
            self.X = X.copy()
        # add additional data I which represents the interception
        self.X = np.hstack((self.X, np.ones((N, 1)) ))
        self.mu = np.append(self.mu, 0.0)
        self.std = np.append(self.std, 1.0)
        self.Y = Y.copy()
        self.beta = np.zeros(m + 1)
    
    def regress(self):
        U = np.linalg.inv(np.dot(self.X.T, self.X))
        V = np.dot(self.X.T, self.Y)
        beta0 = np.dot(U, V)
        # beta0 is regressed on the normalized data, if normalized, we recalculate the real beta
        self.beta = beta0 / self.std
        self.beta[-1] -= (beta0 * self.mu / self.std).sum()
    
    def predict(self, X):
        if len(X.shape) != 2:
            raise ValueError("LS_Matrix predict failed, input X must be numpy ndarray!")
        X = np.hstack((X, np.ones((X.shape[0], 1)) ))
        if X.shape[0] != self.X.shape[0] or X.shape[1] != self.X.shape[1]:
            raise ValueError("LS_Matrix predict failed, input X and training set X have different data size!")
        return np.dot(X, self.beta)
    
    def coefficient(self):
        return self.beta

class LS_Ortho:
    # This method is Algorithm 3.1, page 54 of ESL book, it regresses each Z_i to Y, where Z_i is the residual vector of X_i 
    # by removing correlated parts to X_0, X_1, X_2, ..., X_i - 1, the coefficient i is beta_i = <Z_i, Y> / <Z_i, Z_i>
    def __init__(self, X, Y):
        if len(X.shape) != 2:
            raise ValueError("LS_Matrix init failed, input X must be numpy ndarray!")
        if len(Y.shape) != 1:
            raise ValueError("LS_Matrix init failed, input Y must be numpy 1d array!")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("LS_Matrix init failed, input X and Y must have the same number of rows!")
        self.X = X.copy()
        self.Y = Y.copy()
        self.beta = np.zeros(X.shape[1] + 1)
    
    def regress(self):
        N = self.X.shape[0]
        m = self.X.shape[1]
        beta0 = np.zeros(m + 1)
        # this is the decompose matrix D, basically Z = np.dot([I,X], D)
        D = np.zeros((m + 1, m + 1))
        Z = np.zeros((N, m + 1))
        curZ = np.ones(N)
        Z[:, 0] = curZ
        beta0[0] = (curZ * self.Y).sum() / (curZ * curZ).sum()
        D[0,0] = 1
        for i in range(0, m):
            curZ = self.X[:, i]
            alpha = np.zeros(m + 1)
            for j in range(0, i + 1):
                alpha[j] = -(curZ * Z[:,j]).sum() / (Z[:,j] * Z[:,j]).sum()
                curZ += alpha[j] * Z[:, j]
            # curX is the residual Z_i, now calculate beta_i
            Z[:, i + 1] = curZ
            beta0[i + 1] = (curZ * self.Y).sum() / (curZ * curZ).sum()
            # and update D
            D[:, i + 1] = np.dot(D, alpha)
            D[i + 1, i + 1] = 1.0
        # now we have y_pred = np.dot(Z, beta0), since Z = np.dot([I,X], D), beta_X = np.dot(D, beta0)
        beta_X = np.dot(D, beta0)
        self.beta[-1] = beta_X[0]
        self.beta[:-1] = beta_X[1:]
    
    def predict(self, X):
        if len(X.shape) != 2:
            raise ValueError("LS_Matrix predict failed, input X must be numpy ndarray!")
        if X.shape[0] != self.X.shape[0] or X.shape[1] != self.X.shape[1]:
            raise ValueError("LS_Matrix predict failed, input X and training set X have different data size!")
        X = np.hstack((X, np.ones((X.shape[0], 1)) ))
        return np.dot(X, self.beta)
    
    def coefficient(self):
        return self.beta           
                
class LS_Ridge:
    # Ridge regularized linear regression, we will use gradient method to solve for beta
    def __init__(self, X, Y, w):
        if len(X.shape) != 2:
            raise ValueError("LS_Matrix init failed, input X must be numpy ndarray!")
        if len(Y.shape) != 1:
            raise ValueError("LS_Matrix init failed, input Y must be numpy 1d array!")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("LS_Matrix init failed, input X and Y must have the same number of rows!")
        # for ridge regression we always do normalization for stability of the gradient method
        self.muX = np.mean(X, axis = 0)
        self.stdX = np.std(X, axis = 0)
        self.X = (X - self.muX) / self.stdX
        self.muY = np.mean(Y)
        self.stdY = np.std(Y)
        self.Y = (Y - self.muY) / self.stdY
        self.w = w
        self.beta = np.zeros(X.shape[1] + 1)
    
    def deriv(self, beta):
        # given beta, we calculate the first derivative of the loss function
        # loss function is defined as ||Y - X.dot(beta)||^2 + ||w.dot(beta)||^2
        # first derivative is -2X.T.dot(Y-X.dot(beta)) + 2w*beta
        return -2 * self.X.T.dot(self.Y - self.X.dot(beta)) + 2*self.w*beta

    def regress(self):
        guess = self.beta[:-1].copy()
        beta0 = MathUtils.gradient(self, guess)
        # above beta0 is the regressed results for normalized X and Y, i.e. (Y-muY) / stdY = ((X-muX)/stdX).dot(beta0)
        # therefore we have beta[:-1] = beta0 * stdY / stdX, beta[-1] = beta0 * muX * stdY / stdX + muY
        self.beta[:-1] = beta0 * self.stdY / self.stdX
        self.beta[-1] = self.muY -(beta0 * self.muX * self.stdY / self.stdX).sum()
    
    def predict(self, X):
        if len(X.shape) != 2:
            raise ValueError("LS_Matrix predict failed, input X must be numpy ndarray!")
        if X.shape[0] != self.X.shape[0] or X.shape[1] != self.X.shape[1]:
            raise ValueError("LS_Matrix predict failed, input X and training set X have different data size!")
        X = np.hstack((X, np.ones((X.shape[0], 1)) ))
        return np.dot(X, self.beta)
    
    def coefficient(self):
        return self.beta
    
class LS_Lasso:
    # Ridge regularized linear regression, we will use gradient method to solve for beta
    def __init__(self, X, Y, w):
        if len(X.shape) != 2:
            raise ValueError("LS_Matrix init failed, input X must be numpy ndarray!")
        if len(Y.shape) != 1:
            raise ValueError("LS_Matrix init failed, input Y must be numpy 1d array!")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("LS_Matrix init failed, input X and Y must have the same number of rows!")
        # for ridge regression we always do normalization for stability of the gradient method
        self.muX = np.mean(X, axis = 0)
        self.stdX = np.std(X, axis = 0)
        self.X = (X - self.muX) / self.stdX
        self.muY = np.mean(Y)
        self.stdY = np.std(Y)
        self.Y = (Y - self.muY) / self.stdY
        self.w = w
        self.beta = np.zeros(X.shape[1] + 1)
    
    def deriv(self, beta):
        # given beta, we calculate the first derivative of the loss function
        # loss function is defined as ||Y - X.dot(beta)||^2 + |w.dot(beta)|
        # first derivative is -2X.T.dot(Y-X.dot(beta)) + w*sign(beta)
        return -2 * self.X.T.dot(self.Y - self.X.dot(beta)) + 2*self.w*np.sign(beta)

    def regress(self):
        guess = self.beta[:-1].copy()
        beta0 = MathUtils.gradient(self, guess)
        # above beta0 is the regressed results for normalized X and Y, i.e. (Y-muY) / stdY = ((X-muX)/stdX).dot(beta0)
        # therefore we have beta[:-1] = beta0 * stdY / stdX, beta[-1] = beta0 * muX * stdY / stdX + muY
        self.beta[:-1] = beta0 * self.stdY / self.stdX
        self.beta[-1] = self.muY -(beta0 * self.muX * self.stdY / self.stdX).sum()
    
    def predict(self, X):
        if len(X.shape) != 2:
            raise ValueError("LS_Matrix predict failed, input X must be numpy ndarray!")
        if X.shape[0] != self.X.shape[0] or X.shape[1] != self.X.shape[1]:
            raise ValueError("LS_Matrix predict failed, input X and training set X have different data size!")
        X = np.hstack((X, np.ones((X.shape[0], 1)) ))
        return np.dot(X, self.beta)
    
    def coefficient(self):
        return self.beta
    
class LS_LARS:
    # least angle regression, this one is very complicated in math
    # the idea is as following: starting from orthogonal method, let's assume that all X vectors are orthogonal at the begining
    # then it is clear that regression is just to project Y onto each X_i. However, if they are not orthogonal we cannot do so
    # LARS is to go along this apporach, in the sense that we do not fully project Y onto X_0, given that X_0 is the most correlated X_i
    # instead, we partially project Y to X_0 so that its remenant correlation with X_0 is equal to X_1, then we project along X_0 + X_1 (assume all X_i are unit vectors)
    # it is obvious that X_0 + X_1 has the same angle to X_0 and X_1, and we repeat this process until finish. The equal angle vector is where this name LARS comes from
    # interesting thing is that if we early stop, we will get Lasso results
    # see "Least Angle Regression", the Annals of Statistics 2004 407-499
    # and https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf
    def __init__(self, X, Y):
        if len(X.shape) != 2:
            raise ValueError("LS_Matrix init failed, input X must be numpy ndarray!")
        if len(Y.shape) != 1:
            raise ValueError("LS_Matrix init failed, input Y must be numpy 1d array!")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("LS_Matrix init failed, input X and Y must have the same number of rows!")
        # as usual we normalize inputs
        self.muX = np.mean(X, axis = 0)
        self.stdX = np.std(X, axis = 0)
        self.X = (X - self.muX) / self.stdX
        self.muY = np.mean(Y)
        self.stdY = np.std(Y)
        self.Y = (Y - self.muY) / self.stdY
        self.beta = np.zeros(X.shape[1] + 1)
    
    def regress(self):
        # WARNING: this is based on my current understanding, it is not necessarily correct, especially currently we only have 1D test cases
        # TODO: use high dimension data, compare with scipy LARS results
        N = self.X.shape[0]
        m = self.X.shape[1]
        selected = set()
        signs = np.zeros(m)
        beta0 = np.zeros(m)
        # Y_hat is current prediction
        Y_hat = np.zeros(N)
        # C is the correlation vector, because we have normalized our data so E(X_i) = E(Y) = 0, we have simply C = X.T.dot(y - y_hat)
        C = self.X.T.dot(self.Y - Y_hat)
        next_j = np.argmax(np.abs(C))
        selected.add(next_j)
        signs[next_j] = np.sign(C[next_j])
        # we start adding vectors now
        for i in range(m):
            cur_index = list(selected)
            X = self.X[:,cur_index] * signs[cur_index]
            G = X.T.dot(X)
            G_inv = np.linalg.inv(G)
            G_inv_row_sum = G_inv.sum(axis=1)
            A = np.sqrt(np.sum(G_inv_row_sum))
            w = A * G_inv_row_sum
            u = X.dot(w)
            a = X.T.dot(u)
            gamma = None
            max_cor = np.abs(C).max()
            if i < m - 1:
                # we need to choose the next X vector to add
                next_j = None
                next_sign = 0
                for j in range(m):
                    if j in selected:
                        continue
                    v0 = (max_cor - C[j]) / (A - a[j])
                    v1 = (max_cor + C[j]) / (A + a[j])
                    if v0 > 0 and (gamma is None or v0 < gamma):
                        gamma = v0
                        next_j = j
                        next_sign = 1
                    if v1 > 0 and (gamma is None or v1 < gamma):
                        gamma = v1
                        next_j = j
                        next_sign = -1
                selected.add(next_j)
                signs[next_j] = next_sign
            else:
                # otherwise we already have every vectors in set
                gamma = max_cor / A
            # update beta0 and C
            for index in cur_index:
                beta0[index] += signs[index] * gamma * w[index]
            Y_hat = X.dot(beta0)
            C = self.X.T.dot(self.Y - Y_hat)
        # set final regressed beta, use similar method used in Ridge and Lasso when normalization is performed
        self.beta[:-1] = beta0 * self.stdY / self.stdX
        self.beta[-1] = self.muY -(beta0 * self.muX * self.stdY / self.stdX).sum()
        
    def predict(self, X):
        if len(X.shape) != 2:
            raise ValueError("LS_Matrix predict failed, input X must be numpy ndarray!")
        if X.shape[0] != self.X.shape[0] or X.shape[1] != self.X.shape[1]:
            raise ValueError("LS_Matrix predict failed, input X and training set X have different data size!")
        X = np.hstack((X, np.ones((X.shape[0], 1)) ))
        return np.dot(X, self.beta)
    
    def coefficient(self):
        return self.beta
        
        
        
    
    
    
    
    
    
    
    
    
    