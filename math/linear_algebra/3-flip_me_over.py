#!/usr/bin/env python3
"""_summary_
Contains a func matrix_transpose(matrix) that returns the transpose of a matrix
"""


def matrix_transpose(matrix):
    """_summary_

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: A list of lists
    """
    new_matrix = [
        [0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))
    ]
    for r in range(len(matrix)):
        for c in range(len(matrix[r])):
            new_matrix[c][r] = matrix[r][c]
    return new_matrix
