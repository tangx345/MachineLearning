# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:48:35 2021

@author: Yang
"""

import numpy as np

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
                
                
                
                
                
                
                
                