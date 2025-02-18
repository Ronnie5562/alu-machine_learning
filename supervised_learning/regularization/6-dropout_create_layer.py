#!/usr/bin/env python3
"""
    A function def dropout_create_layer(prev, n, activation, keep_prob):
    that creates a layer of a neural network using dropout:
"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout

    Args:
        - prev: tensor containing the output of the previous layer
        - n: number of nodes the new layer should contain
        - activation: activation function that should be used on the layer
        - keep_prob: probability that a node will be kept

    Returns:
        - the tensor output of the new layer
    """
    # Initialize weights with He initialization
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=initializer
    )
    output = layer(prev)

    # Apply dropout
    layer = tf.layers.dropout(output, rate=1 - keep_prob)

    return layer
