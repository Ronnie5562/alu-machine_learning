#!/usr/bin/env python3
"""
    A function def dropout_forward_prop(X, weights, L, keep_prob):
    that conducts forward propagation using Dropout:
"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        - X: numpy.ndarray of shape (nx, m) containing the input data
        for the network
        - weights: dictionary of the weights and biases of the neural network
        - L: number of layers in the network
        - keep_prob: probability that a node will be kept
        - keep_prob: probability that a node will be kept for dropout

    Returns:
        - a dictionary containing the outputs of each layer and
        the dropout mask used on each layer
    """
    cache = {}
    cache["A0"] = X
    for i in range(L):
        W = weights["W" + str(i + 1)]
        b = weights["b" + str(i + 1)]
        A = cache["A" + str(i)]
        Z = np.matmul(W, A) + b

        if i == L - 1:
            # Softmax activation for the output layer
            t = np.exp(Z)
            cache["A" + str(i + 1)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            # Tanh activation for hidden layers
            A = np.tanh(Z)

            # Dropout mask
            D = ((np.random.rand(A.shape[0], A.shape[1])
                  < keep_prob).astype(int))
            A *= D
            A /= keep_prob
            cache["A" + str(i + 1)] = A
            cache["D" + str(i + 1)] = D

    return cache
