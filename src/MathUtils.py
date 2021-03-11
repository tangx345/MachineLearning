# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 19:09:23 2021

@author: Yang
"""

import numpy as np

def gradient(f, guess, step = 0.01, maxIter = 1e4, tol = 1e-9):
    iterNum = 0
    deriv = f.deriv(guess)
    while iterNum < maxIter and np.abs(deriv).max() > tol:
        guess -= step * deriv
        deriv = f.deriv(guess)
        iterNum += 1
        if iterNum >= 5e3:
            # for large iteration numbers, it could be that the step is too large, we try smaller step
            step = 1e-4
    return guess