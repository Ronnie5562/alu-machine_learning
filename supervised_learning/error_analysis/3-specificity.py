#!/usr/bin/env python3
"""
    A function def specificity(confusion):
    that calculates the specificity for each
    class in a confusion matrix:
"""


import numpy as np


def specificity(confusion):
    """
    A function def specificity(confusion):
    that calculates the specificity for each
    class in a confusion matrix:

    Args:
        - confusion is a confusion numpy.ndarray of shape
        (classes, classes) where row indices
        represent the correct labels and column
        indices represent the predicted labels
         - classes is the number of classes

    Returns:
        - a numpy.ndarray of shape (classes,) containing
        the specificity of each class
    """
    classes = confusion.shape[0]
    specificity_values = np.zeros(classes)

    for i in range(classes):
        true_negatives = np.sum(np.delete(
            np.delete(confusion, i, axis=0), i, axis=1)
        )
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]
        specificity_values[i] = (
            true_negatives / (true_negatives + false_positives)
        )

    return specificity_values
