#!/usr/bin/env python3
"""
    Function def sensitivity(confusion):
    that calculates the sensitivity for
    each class in a confusion matrix:
"""


import numpy as np


def sensitivity(confusion):
    """
    That calculates the sensitivity for each class
    in a confusion matrix:

    Args:
        - confusion is a confusion numpy.ndarray of shape
        (classes, classes) where row indices
        represent the correct labels and column
        indices represent the predicted labels
         - classes is the number of classes

    Returns:
        - a numpy.ndarray of shape (classes,) containing
        the sensitivity of each class
    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
