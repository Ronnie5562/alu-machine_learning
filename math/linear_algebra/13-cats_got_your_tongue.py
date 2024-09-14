#!/usr/bin/env python3
"""_summary_
Contains a function that concatenates matrices on different axis
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    return np.concatenate((mat1, mat2), axis=axis)
