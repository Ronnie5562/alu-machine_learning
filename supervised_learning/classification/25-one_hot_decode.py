#!/usr/bin/env python3
"""
    defines function that converts a numeric label vector
    into a one-hot matrix
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    one hot decode function
    Args:
        one_hot: one-hot encoded numpy.ndarray with shape (classes, m)
        classes: number of classes
    Returns:
        numpy.ndarray with shape (m,) containing the numeric labels for each
        example, or None on failure
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    vector = one_hot.transpose().argmax(axis=1)
    return vector
