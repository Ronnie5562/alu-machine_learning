#!/usr/bin/env python3
"""
    The function def create_confusion_matrix(labels, logits):
    that creates a confusion matrix:
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    A function that creates a confusion matrix:

    Args:
        - labels is a one-hot numpy.ndarray of shape (m, classes)
        containing the correct labels for each data point
        - m is the number of data points
        - classes is the number of classes
        - logits is a one-hot numpy.ndarray of shape (m, classes)
        containing the predicted labels

    Returns:
        - a confusion matrix
            - (classes, classes) with row indices representing the correct
            labels and column indices representing the predicted labels
    """
    return np.matmul(labels.T, logits)
