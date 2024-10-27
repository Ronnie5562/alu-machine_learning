#!/usr/bin/env python3
"""
    A function def convolve_grayscale_same(images, kernel):
    that performs a same convolution on grayscale images:
"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    '''
        A function def convolve_grayscale_same(images, kernel):

        Args:
            images is a numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            kernel is a numpy.ndarray with shape (kh, kw)
            containing the kernel for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        Returns:
            a numpy.ndarray containing
            the convolved images

    '''
    m = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    if kh % 2 == 1:
        ph = (kh - 1) // 2
    else:
        ph = kh // 2
    if kw % 2 == 1:
        pw = (kw - 1) // 2
    else:
        pw = kw // 2
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)
    convoluted = np.zeros((m, height, width))
    for h in range(height):
        for w in range(width):
            output = np.sum(images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
