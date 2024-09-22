#!/usr/bin/env python3
"""__summary__
This file contains the implementation to get the definiteness of a matrix.
"""
import numpy as np


def definiteness(matrix):
    """
    Comp the definiteness of a matrix.

    Args:
        matrix(numpy.ndarray): The matrix whose definiteness is to be computed.

    Returns:
        str or None :[
            "Positive definite",
            "Positive semi-definite",
            "Negative semi-definite",
            "Negative definite",
            "Indefinite",
            None
        ]
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Ensure the matrix is square (n x n)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if all(eigenvalues > 0):
        return "Positive definite"
    elif all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif all(eigenvalues < 0):
        return "Negative definite"
    elif all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif any(eigenvalues > 0) and any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
