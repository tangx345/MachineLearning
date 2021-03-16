# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:22:27 2021

@author: Yang
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('./MathUtils.py')
sys.path.append('./LinearRegression.py')
import MathUtils

class TreeNode:
    def __init__(self, colName, splitValue, predictionValue, left = None, right = None):
        self.colName = colName
        self.splitValue = splitValue
        self.predictionValue = predictionValue
        self.left = left
        self.right = right
    
    def isLeaf(self):
        return self.left is None and self.right is None
    
    def getColName(self):
        return self.colName
    
    def nextNode(self, value):
        if value <= self.splitValue:
            return self.left
        else:
            return self.right
    
    def getPredictionValue(self):
        return self.predictionValue

class Splitter:
    def __init__(self, method = 'Gini', minSubsetSize = 5):
        # supported methods are 'Variance' : find the minimum the total variance of each subset after split, this is most useful when target variable is quantitative
        #                       'Entropy' : find the smallest entropy after split, this is most useful when target variable is categorical
        #                       'Gini' : most popular way to split categorical target variable
        #                       'ChiSquare' : also works with categorical target variable
        # minSubsetSize is the minimum size of subsets after split, this is to avoid overfitting by too small subsets
        self.method = method
        self.minSize = minSubsetSize
    
    def split(self, data):
        # data is the dataset, required to have two columns, first column is the splitter column, second column is the target variable
        # returns the boundary value
        columns = data.columns.values
        if len(columns) != 2:
            raise ValueError("splitter::splitNum: input data must have two columns!")
        if len(data) < 2 * self.minSize:
            # after splitting there is always a subset's length smaller than minSubsetSize
            return np.nan
        data1 = data.sort_values(by=columns[0])
        candidates = data1[columns[0]].unique()
        targets = data1[columns[1]].unique()
        if self.method == 'Variance':
            total_var = None
            split_val = np.nan
            for cand in candidates:
                s1 = data1[columns[0]] <= cand
                s2 = data1[columns[0]] > cand
                if s1.sum() < self.minSize or s2.sum() < self.minSize:
                    continue
                else:
                    var1 = np.var(data1.loc[s1, columns[1]]) + np.var(data1.loc[s2, columns[1]])
                    if total_var is None or var1 < total_var:
                        total_var = var1
                        split_val = cand
            return split_val
        elif self.method == 'Entropy':
            total_entropy = None
            split_val = np.nan
            for cand in candidates:
                s1 = data1[columns[0]] <= cand
                s2 = data1[columns[0]] > cand
                if s1.sum() < self.minSize or s2.sum() < self.minSize:
                    continue
                else:
                    entropy = 1
                    subset1 = data1.loc[s1, columns[1]]
                    subset2 = data1.loc[s2, columns[1]]
                    for target in targets:
                        p1 = (subset1 == target).sum() / len(subset1)
                        p2 = (subset2 == target).sum() / len(subset2)
                        entropy += -p1 * np.log(p1) - p2 * np.log(p2)
                    if total_entropy is None or entropy < total_entropy:
                        total_entropy = entropy
                        split_val = cand
            return split_val
        elif self.method == 'Gini':
            best_gini = None
            split_val = np.nan
            for cand in candidates:
                s1 = data1[columns[0]] <= cand
                s2 = data1[columns[0]] > cand
                if s1.sum() < self.minSize or s2.sum() < self.minSize:
                    continue
                else:
                    gini = 1
                    subset1 = data1.loc[s1, columns[1]]
                    subset2 = data1.loc[s2, columns[1]]
                    for target in targets:
                        p1 = (subset1 == target).sum() / len(subset1)
                        p2 = (subset2 == target).sum() / len(subset2)
                        gini -= p1 * p1 + p2 * p2
                    if best_gini is None or gini < best_gini:
                        best_gini = gini
                        split_val = cand
            return split_val
        elif self.method == 'ChiSquare':
            best_chi_2 = None
            split_val = np.nan
            p_map = {}
            for target in targets:
                p_map[target] = (data1[columns[1]] == target).sum() / len(data1)
            for cand in candidates:
                s1 = data1[columns[0]] <= cand
                s2 = data1[columns[0]] > cand
                if s1.sum() < self.minSize or s2.sum() < self.minSize:
                    continue
                else:
                    chi_2 = 0
                    subset1 = data1.loc[s1, columns[1]]
                    subset2 = data1.loc[s2, columns[1]]
                    for target in targets:
                        act1 = (subset1 == target).sum()
                        exp1 = p_map[target] * len(subset1)
                        act2 = (subset2 == target).sum()
                        exp2 = p_map[target] * len(subset2)
                        chi_2 += np.sqrt((act1 - exp1)**2 / exp1) + np.sqrt((act2 - exp2)**2 / exp2)
                    if best_chi_2 is None or chi_2 < best_chi_2:
                        best_chi_2 = chi_2
                        split_val = cand
            return split_val
        else:
            raise ValueError("splitter::splitNum: unrecongnized splitting method!")

class BinaryTree:
    def __init__(self, splitMethod, minSubsetSize, maxTreeLevel, isCategorical = True, guessWithTie = True):
        self.splitter = Splitter(splitMethod, minSubsetSize)
        self.maxTreeLevel = maxTreeLevel
        self.treeHead = None
        self.isCategorical = isCategorical
        self.guessWithTie = guessWithTie
        self.targetCol = ''
    
    def buildTree(self, data, targetCol):
        columns = set(data.columns.values)
        columns.remove(targetCol)
        self.targetCol = targetCol
        self.treeHead = self.growTree(data, columns, 0)
    
    def growTree(self, data, columns, treeLevel):
        if treeLevel > self.maxTreeLevel:
            predVal = np.nan
            if self.isCategorical:
                values, counts = np.unique(data[self.targetCol].values, return_counts = True)
                frequents = values[counts == counts.max()]
                if len(frequents) > 1:
                    if self.guessWithTie:
                        predVal = MathUtils.randomChoose(frequents)
                    else:
                        raise ValueError("BinaryTree::growTree: fail to grow tree because subset is not splittable and subset has multiple targets with highest frequency!")
                else:
                    predVal = values[np.argmax(counts)]
            else:
                predVal = data[self.targetCol].mean()
            return TreeNode('', np.nan, predVal)
        while len(columns) > 0:
            col = MathUtils.randomChoose(columns)
            split_val = self.splitter.split(data)
            if np.isnan(split_val):
                columns.remove(col)
            else:
                data_l = data.loc[data[col] <= split_val]
                data_2 = data.loc[data[col] > split_val]
                left_child = self.growTree(data_l, columns.copy(), treeLevel + 1)
                right_child = self.growTree(data_2, columns.copy(), treeLevel + 1)
                return TreeNode(col, split_val, np.nan, left_child, right_child)
        # nothing left to split, just return a leaf node
        predVal = np.nan
        if self.isCategorical:
            values, counts = np.unique(data[self.targetCol].values, return_counts = True)
            frequents = values[counts == counts.max()]
            if len(frequents) > 1:
                if self.guessWithTie:
                    predVal = MathUtils.randomChoose(frequents)
                else:
                    raise ValueError("BinaryTree::growTree: fail to grow tree because subset is not splittable and subset has multiple targets with highest frequency!")
            else:
                predVal = values[np.argmax(counts)]
        else:
            predVal = data[self.targetCol].mean()
        return TreeNode('', np.nan, predVal)
    
    def predict(self, X):
        if self.treeHead is None:
            raise ValueError("BinaryTree: prediction failed because tree is not built!")
        y = np.zeros(len(X))
        for ii in range(len(X)):
            curNode = self.treeHead
            while not curNode.isLeaf():
                col = curNode.getColName()
                if col not in X.columns.values:
                    raise ValueError("BinaryTree: prediction failed because column " + curNode.colName() + " is not in predictor data's columns!")
                else:
                    curNode = curNode.nextNode(X[col].iloc[ii])
            y[ii] = curNode.getPredictionValue()
        return y
    
    
    
    
    
    
    