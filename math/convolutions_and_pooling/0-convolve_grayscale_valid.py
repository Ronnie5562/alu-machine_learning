#!/usr/bin/env python3
"""
    A function def convolve_grayscale_valid
    convolve_grayscale_valid(images, kernel)
"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    images is a numpy.ndarray with shape (i, y, x)
    containing multiple grayscale images
    m is the number of images
    y is the height in pixels of the images
    x is the width in pixels of the images
    kernel is a numpy.ndarray with shape (m, n)
    containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    Returns: a numpy.ndarray containing
    the convolved images
    """
    m, n = kernel.shape
    if m == n:
        i, y, x = images.shape
        y = y - m + 1
        x = x - m + 1
        convolved_image = np.zeros((i, y, x))
        for i in range(y):
            for j in range(x):
                shadow_area = images[:, i:i + m, j:j + n]
                convolved_image[:, i, j] = \
                    np.sum(shadow_area * kernel, axis=(1, 2))
    return convolved_image
