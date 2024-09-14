#!/usr/bin/env python3
"""
    Contains a func matrix_shape(matrix) that returns the shape of a matrix
"""


def matrix_shape(matrix):
    """_summary_

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: A list of integers
    """
    if type(matrix) is not list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
