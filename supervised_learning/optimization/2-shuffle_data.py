#!/usr/bin/env python3
'''
    Function def shuffle_data(X, Y): that shuffles
    the data points in two matrices the same way
'''


import numpy as np


def shuffle_data(X, Y):
    '''
        The shuffled X and Y matrices

        Args:
            - X is the first numpy.ndarray of shape (m, nx) to shuffle
                m is the number of data points
                nx is the number of features in X
            - Y is the second numpy.ndarray of shape (m, ny) to shuffle
                m is the same number of data points as in X
                ny is the number of features in Y
    '''
    m = X.shape[0]
    shuffled_indexes = np.random.permutation(m)
    X_shuffled = X[shuffled_indexes]
    Y_shuffled = Y[shuffled_indexes]
    return (X_shuffled, Y_shuffled)
