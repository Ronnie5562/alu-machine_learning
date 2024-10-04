#!/usr/bin/env python3
"""
This file contains the implementation of summation_i_squared
"""


def summation_i_squared(n):
    """_summary_
    A recursive function that computes the result a sigma notation.
    """
    if not int(n):
        return None
    if n == 1:
        return 1

    return n**2 + summation_i_squared(n-1)
