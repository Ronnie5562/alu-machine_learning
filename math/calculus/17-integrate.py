#!/usr/bin/env python3
"""_summary_
This file contains the implementation of poly_integral
"""


def poly_integral(poly, C=0):
    """_summary_
    Computes the coefficients of the terms in the
    integral of a function using Sum rule
    """
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    integrals = [C]
    for term in enumerate(poly):
        if term[0] == 0:
            integrals.append(term[1])
        else:
            integrals.append(term[1] / (term[0] + 1))

    return integrals
