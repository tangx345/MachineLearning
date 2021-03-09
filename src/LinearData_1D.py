# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:55:53 2021

@author: Yang
"""

import numpy as np
import pandas as pd
from typing import List

dataFolder = "../data/"

class Generator1D_WhiteNoise:
    # This class creates 1D linear data [X, Y] in pandas dataframe, and save it in ../data/ folder
    def __init__(self, beta : List[float], N : int, xRange : List[float], noiseSigma : float, filename : str):
        if len(beta) != 2:
            raise ValueError("Coefficient beta for 1D linear data generator must be of size 2!")
        if len(xRange) != 2:
            raise ValueError("Data range for X must be of size 2!")
        # inputs: beta is the coefficient Y = beta * [X, I] = a * X + b
        #         N is the length of data
        #         xRange is a range [a, b] so that the generated X will be uniformly distributed on [a, b]
        #         sigma is the vol of noise
        #         fn is the filename to be saved as
        self.beta = beta
        self.N = N
        self.xRange = xRange
        self.sigma = noiseSigma
        self.fn = filename
    
    def generate(self):
        # generate N uniformly distributed X
        X = np.random.uniform(self.xRange[0], self.xRange[1], self.N)
        noise = np.random.normal(0.0, self.sigma, self.N)
        Y = self.beta[0] * X + self.beta[1] + noise
        data = pd.DataFrame()
        data['X'] = X
        data['Y'] = Y
        data.to_csv(dataFolder + self.fn + '.csv', index = False)
        