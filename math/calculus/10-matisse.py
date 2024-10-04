#!/usr/bin/env python3
"""_summary_
This file contains the implementation of poly_derivative
"""


def poly_derivative(poly):
    """_summary_
    Computes the coefficients of the terms in the
    derivative of a function using Sum rule
    """
    if not isinstance(poly, list) or len(poly) <= 1:
        return [0]

    derivatives = []
    for term in enumerate(poly):
        if term[0] == 0:
            continue
        derivatives.append(term[0] * term[1])

    return derivatives
