#!/usr/bin/env python3
"""
    function def normalization_constants(X): that calculates the normalization
    (standardization) constants of a matrix:
"""


import numpy as np


def normalization_constants(X):
    '''
    X is the numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features
    '''
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
