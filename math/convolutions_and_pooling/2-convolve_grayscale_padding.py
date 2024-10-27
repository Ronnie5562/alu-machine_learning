#!/usr/bin/env python3
'''
    A function def convolve_grayscale_padding(images, kernel, padding):
    that performs a same convolution on grayscale images:
'''


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''
        A function def convolve_grayscale_padding(images, kernel, padding):

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
            padding is a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        Returns:
            a numpy.ndarray containing
            the convolved images
    '''
    m = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)
    ch = height + (2 * ph) - kh + 1
    cw = width + (2 * pw) - kw + 1
    convoluted = np.zeros((m, ch, cw))
    for h in range(ch):
        for w in range(cw):
            output = np.sum(images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
