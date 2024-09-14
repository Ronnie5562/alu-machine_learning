#!/usr/bin/env python3

def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    output_matrix = []

    for i in range(len(mat1)):
        current_row = [
            mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))
        ]

        output_matrix.append(current_row)

    return output_matrix
