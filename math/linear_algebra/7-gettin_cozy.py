#!/usr/bin/env python3
"""_summary_
Contains a func cat_matrices2D() that concatenates two matrices
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
