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
            self.X = X
        # add additional data I which represents the interception
        self.X = np.hstack((self.X, np.ones((N, 1)) ))
        self.mu = np.append(self.mu, 0.0)
        self.std = np.append(self.std, 1.0)
        self.Y = Y
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