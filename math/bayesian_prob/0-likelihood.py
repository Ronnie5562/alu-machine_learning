#!/usr/bin/env python3
"""
def likelihood(x, n, P): that calculates the likelihood
of obtaining this data given various hypothetical
probabilities of developing severe side effects
"""


import numpy as np


def likelihood(x, n, P):
    """
    Args:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray of length equal to the
        number of patients that develop severe side effects

    Returns:
        the likelihood of obtaining x and n
    """
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is an integer and greater than or equal to 0
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    # check if x is greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if p is a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if all values in p are in the range [0, 1]
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the combination using np.math.factorial
    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(n - x) * factorial(x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
