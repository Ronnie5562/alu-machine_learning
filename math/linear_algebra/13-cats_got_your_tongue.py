#!/usr/bin/env python3
"""_summary_
Contains a function that concatenates matrices on different axis
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    return np.concatenate((mat1, mat2), axis=axis)
