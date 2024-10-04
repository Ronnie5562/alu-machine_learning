#!/usr/bin/env python3
"""_summary_
This file contains the implementation of poly_derivative
"""


def poly_derivative(poly):
    """_summary_
    Computes the coefficients of the terms in the
    derivative of a function using Sum rule
    """
    if not isinstance(poly, list) or not poly:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 1:
        return [0]

    derivatives = []
    for power, coefficient in enumerate(poly):
        if power == 0:
            continue
        derivatives.append(power * coefficient)

    return derivatives
