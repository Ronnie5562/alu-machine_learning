#!/usr/bin/env python3
"""_summary_
Contains a func mat_mul(mat1, mat2) that multiplies two matrices
"""


def mat_mul(mat1, mat2):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(mat1[0]) == len(mat2):
        mat2_T = list(zip(*mat2))

        return [
            [
                sum(i * j for i, j in zip(row, col))
                for col in mat2_T
            ]
            for row in mat1
        ]
    else:
        return None
