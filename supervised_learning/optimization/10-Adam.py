#!/usr/bin/env python3
"""
    Function def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    that creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm:
"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm

    Args:
        - loss is the loss of the network
        - alpha is the learning rate
        - beta1 is the weight used for the first moment
        - beta2 is the weight used for the second moment
        - epsilon is a small number to avoid division by zero

    Returns:
        The Adam optimization operation
    """
    # Initialize the Adam optimizer with the given parameters
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    # Create the training operation by minimizing the loss
    train_op = optimizer.minimize(loss)

    return train_op
