#!/usr/bin/env python3
"""
    A function def precision(confusion):
    that calculates the precision for each class in
    a confusion matrix
"""


import numpy as np


def precision(confusion):
    """
    A function def precision(confusion):
    that calculates the precision for each class in
    a confusion matrix

    Args:
        - confusion is a confusion numpy.ndarray of shape
        (classes, classes) where row indices
        represent the correct labels and column
        indices represent the predicted labels
         - classes is the number of classes

    Returns:
        - a numpy.ndarray of shape (classes,) containing
        the precision of each class
    """
    return np.diag(confusion) / np.sum(confusion, axis=0)
