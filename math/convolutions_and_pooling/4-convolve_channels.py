#!/usr/bin/env python3
"""
    A function def
    convolve(images, kernels, padding='same', stride=(1, 1)):
    that performs a convolution on images using multiple kernels:
"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple channels
    using given padding and stride

    parameters:
        images [numpy.ndarray with shape (m, h, w, c)]:
            contains multiple images
            m: number of images
            h: height in pixels of all images
            w: width in pixels of all images
            c: number of channels in the image
        kernel [numpy.ndarray with shape (kh, kw, c)]:
            contains the kernel for the convolution
            kh: height of the kernel
            kw: width of the kernel
            c: number of channels in the image
        padding [tuple of (ph, pw) or 'same' or 'valid']:
            ph: padding for the height of the images
            pw: padding for the width of the images
            'same' performs same convolution
            'valid' performs valid convolution
        stride [tuple of (sh, sw)]:
            sh: stride for the height of the image
            sw: stride for the width of the image

    if needed, images should be padded with 0s
    function may only use two for loops maximum and no other loops are allowed

    returns:
        numpy.ndarray contained convolved images
    """
    m, height, width, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((((height - 1) * sh) + kh - height) // 2) + 1
        pw = ((((width - 1) * sw) + kw - width) // 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    ch = ((height + (2 * ph) - kh) // sh) + 1
    cw = ((width + (2 * pw) - kw) // sw) + 1
    convoluted = np.zeros((m, ch, cw))

    for i, h in enumerate(range(0, (height + (2 * ph) - kh + 1), sh)):
        for j, w in enumerate(range(0, (width + (2 * pw) - kw + 1), sw)):
            output = np.sum(images[:, h: h + kh, w: w + kw, :] * kernel,
                            axis=(1, 2, 3))
            convoluted[:, i, j] = output

    return convoluted
