#!/usr/bin/env python3
"""
    defines function that converts a numeric label vector
    into a one-hot matrix
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    one hot encode function
    Args:
        Y: numpy.ndarray of shape (classes,)
        classes: number of classes
    Returns:
        one-hot encoding of Y with shape (classes, m)
        m: number of examples
    """
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        one_hot = np.eye(classes)[Y].transpose()
        return one_hot
    except Exception as err:
        return None
