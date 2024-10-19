#!/usr/bin/env python3
'''
    a function def correlation(C):
    that calculates the correlation matrix
    of a data set
'''


import numpy as np


def correlation(C):
    '''
    Args:
        C is a numpy.ndarray of shape (d, d) containing
        the covariance matrix

    Returns:
        Returns a numpy.ndarray of shape (d, d)
        containing the correlation matrix
    '''
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    corr = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            corr[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])

    return corr
