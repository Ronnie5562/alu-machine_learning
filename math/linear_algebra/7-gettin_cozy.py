#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):
    if axis == 0:
        return mat1 + mat2
    return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
