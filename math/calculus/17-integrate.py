#!/usr/bin/env python3
"""_summary_
This file contains the implementation of poly_integral
"""


def poly_integral(poly, C=0):
    """_summary_
    Computes the coefficients of the terms in the
    integral of a function using Sum rule
    """
    if not isinstance(poly, list) or not isinstance(C, (int, float)) or not poly:  # noqa
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    integrals = [C]
    for power, coefficient in enumerate(poly):
        if power == 0:
            integrals.append(coefficient)
        else:
            integral = coefficient / (power + 1)
            integrals.append(
                int(integral) if integral.is_integer() else integral
            )
    while integrals[-1] == 0 and len(integrals) > 1:
        integrals.pop()
    return integrals
