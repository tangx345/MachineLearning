# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:44:23 2021

@author: Yang
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('./MathUtils.py')
import MathUtils

class LDA:
    def __init__(self, df):
        N = len(df)
        m = len(df.columns) - 1
        if m < 1:
            raise ValueError("LDA init failed, input X is empty!")
        x_cols = ['x{}'.format(i) for i in range(m)]
        self.df = pd.DataFrame(df.values, columns = x_cols + ['Y'])
        self.G = np.sort(self.df['Y'].unique())
        K = len(self.G)
        
        self.pi = []
        self.mu = []
        self.covMat = np.zeros((m, m))
        
        for i in range(K):
            cur_X = self.df[self.df['Y'] == self.G[i]][x_cols]
            self.pi.append(len(cur_X) / N)
            self.mu.append(cur_X.mean())
            self.covMat = self.covMat + np.cov(cur_X, rowvar = False)
        self.covMat = self.covMat / K
        self.invCov = np.linalg.inv(self.covMat)
    
    def predict(self, X):
        N = len(X)
        K = len(self.G)
        prob = np.zeros((N, K))
        for i in range(K):
            cur_pi = self.pi[i]
            cur_mu = self.mu[i]
            prob[:, i] = np.dot(X, np.dot(self.invCov, cur_mu)) - 0.5 * np.dot(cur_mu, np.dot(self.invCov, cur_mu)) + np.log(cur_pi)
        index = np.argmax(prob, axis = 1)
        return np.array([self.G[i] for i in index])

class QDA:
    def __init__(self, df):
        N = len(df)
        m = len(df.columns) - 1
        if m < 1:
            raise ValueError("LDA init failed, input X is empty!")
        x_cols = ['x{}'.format(i) for i in range(m)]
        self.df = pd.DataFrame(df.values, columns = x_cols + ['Y'])
        self.G = np.sort(self.df['Y'].unique())
        K = len(self.G)
        
        self.pi = []
        self.mu = []
        self.covMat = []
        self.covDet = []
        self.invCov = []
        
        for i in range(K):
            cur_X = self.df[self.df['Y'] == self.G[i]][x_cols]
            self.pi.append(len(cur_X) / N)
            self.mu.append(cur_X.mean())
            self.covMat.append(np.cov(cur_X, rowvar = False))
            self.covDet.append(np.linalg.det(self.covMat[-1]))
            self.invCov.append(np.linalg.inv(self.covMat[-1]))
    
    def predict(self, X):
        N = len(X)
        K = len(self.G)
        prob = np.zeros((N, K))
        for i in range(K):
            cur_pi = self.pi[i]
            cur_X = X - self.mu[i].values
            prob[:, i] = np.diag(-0.5 * np.log(self.covDet[i]) - 0.5 * np.dot(cur_X, np.dot(self.invCov[i], cur_X.T)) + np.log(cur_pi))
        index = np.argmax(prob, axis = 1)
        return np.array([self.G[i] for i in index])

class Logistic:
    # currently logistic only works for two classes
    def __init__(self, df):
        N = len(df)
        m = len(df.columns) - 1
        if m < 1:
            raise ValueError("LDA init failed, input X is empty!")
        x_cols = ['x{}'.format(i) for i in range(m)]
        self.df = pd.DataFrame(df.values, columns = x_cols + ['Y'])
        self.X = np.hstack((self.df[x_cols].values, np.ones((N, 1)) ))
        self.Y = self.df['Y'].values.reshape((N, 1))
        self.beta = np.zeros((m + 1, 1))
    
    def loss(self, beta):
        e = np.exp(np.dot(self.X, beta))
        p = e / (1 + e)
        return np.dot(self.X.T, self.Y - p)
    
    def deriv(self, beta):
        e = np.exp(np.dot(self.X, beta))
        p = e / (1 + e)
        h = p * (1 - p)
        W = np.diag(h.flatten())
        return -np.dot(self.X.T, np.dot(W, self.X))
    
    def regress(self):
        self.beta = MathUtils.newtonRaphson(self, self.beta)
    
    def predict(self, X):
        N = len(X)
        X = np.hstack((X, np.ones((N, 1))))
        p = np.dot(X, self.beta)
        y_predict = (p > 0).astype(int)
        return y_predict