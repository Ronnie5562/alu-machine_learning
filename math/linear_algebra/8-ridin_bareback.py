#!/usr/bin/env python3

def mat_mul(mat1, mat2):
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
