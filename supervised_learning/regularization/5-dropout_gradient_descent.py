#!/usr/bin/env python3
"""
    function def dropout_gradient_descent(
        Y, weights, cache, alpha, keep_prob, L)
        that updates the weights of a neural network
        with Dropout regularization using gradient descent:
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization

    Args:
        - Y is a one-hot numpy.ndarray of shape (classes, m)
        that contains the correct
        labels for the data
        classes is the number of classes
        - m is the number of data points
        - weights is a dictionary of the weights and biases
        of the neural network
        - cache is a dictionary of the outputs and dropout
        masks of each layer of
        the neural network
        - alpha is the learning rate
        - keep_prob is the probability that a node will be kept
        - L is the number of layers of the network

        - All layers use thetanh activation function except the last,
        which uses
        the softmax activation function

    Returns:
        - The updated weights of the neural Network
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        dW = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if layer > 1:
            D_prev = cache['D' + str(layer - 1)]
            A_prev = cache['A' + str(layer - 1)]
            dz = np.matmul(W.T, dz) * (1 - A_prev ** 2)
            dz *= D_prev  # Apply dropout mask
            dz /= keep_prob  # Scale the activation for the dropped units

        # Update weights and biases
        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db
