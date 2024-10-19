#!/usr/bin/env python3
'''
    a function def intersection(x, n, P, Pr):
    that calculates the intersection of obtaining
    the data with the probability of developing
    severe side effects given the data
'''


import numpy as np


def intersection(x, n, P, Pr):
    """
    Args:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray of length equal to the
        number of patients that develop severe side effects
        Pr is a 1D numpy.ndarray of length equal to the
        number of patients that develop severe side effects

    Returns:
        the probability of obtaining the data
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

    #  check if Pr is not a numpy.ndarray with the same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    for value in range(P.shape[0]):
        if P[value] > 1 or P[value] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[value] > 1 or Pr[value] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")
    # likelihood calculated as binomial distribution
    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(n - x) * factorial(x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))
    # intersection is the likelihood times priors
    intersection = likelihood * Pr

    return intersection
