#!/usr/bin/env python3
"""
    A function def
    convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    that performs a convolution on grayscale images:
"""


import numpy as np


def convolve_grayscale(images, kernel, padding="same", stride=(1, 1)):
    """
    A function def convolve_grayscale(images, kernel,
        padding='same', stride=(1, 1)):
    that performs a convolution on grayscale images:

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
        padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0’s
        stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image

    Returns:
        a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh, sw = stride
    if padding == "same":
        ph = int(((height - 1) * stride[0] + kh - height) / 2) + 1
        pw = int(((width - 1) * stride[1] + kw - width) / 2) + 1
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        ph = padding[0]
        pw = padding[1]
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    "constant", constant_values=0)
    ch = ((height + (2 * ph) - kh) // sh) + 1
    cw = ((width + (2 * pw) - kw) // sw) + 1
    convolved_image = np.zeros((m, ch, cw))

    i = 0
    for h in range(0, (height + (2 * ph) - kh + 1), sh):
        j = 0
        for w in range(0, (width + (2 * pw) - kw + 1), sw):
            output = np.sum(images[:, h:h + kh, w:w + kw] *
                            kernel, axis=1).sum(axis=1)
            convolved_image[:, i, j] = output
            j += 1
        i += 1
    return convolved_image
