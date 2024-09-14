#!/usr/bin/env python3

def matrix_transpose(matrix):
    new_matrix = [
        [0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))
    ]
    for r in range(len(matrix)):
        for c in range(len(matrix[r])):
            new_matrix[c][r] = matrix[r][c]
    return new_matrix
