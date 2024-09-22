#!/usr/bin/env python3
"""__summary__
This file contains the implementation to compute the determinant of a matrix.
"""


def determinant(matrix):
    """Compute the determinant of a matrix.

    Args:
        matrix (list of lists): The matrix for which to compute its det.

    Returns:
        float: The determinant of the matrix.

    Raises:
        TypeError: If the matrix is not a list of lists.
        ValueError: If the matrix is not square or is invalid.
    """

    # Validate if matrix is a list of lists
    if not isinstance(matrix, list) or \
        len(matrix) == 0 or \
            not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    # Handle 0x0 matrix
    if (len(matrix) == 1 and len(matrix[0]) == 0):
        return 1

    # Check if the matrix is square
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError('matrix must be a square matrix')

    def compute_determinant(matrix):
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        elif n == 2:
            return (
                matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        else:
            det = 0
            for col in range(n):
                sub_matrix = [row[:col] + row[col+1:] for row in matrix[1:]]
                cofactor = ((-1) ** col) * \
                    matrix[0][col] * compute_determinant(sub_matrix)
                det += cofactor
            return det

    return compute_determinant(matrix)
