#!/usr/bin/env python3
"""_summary_
Contains a func np_elementwise(mat1, mat2) that performs
element-wise operations on two matrices
"""


def np_elementwise(mat1, mat2):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2]
