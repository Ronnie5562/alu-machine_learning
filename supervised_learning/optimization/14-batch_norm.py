#!/usr/bin/env python3
"""
    function def create_batch_norm_layer(prev, n, activation):
    that creates a batch normalization layer for a neural network
"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for
    a neural network in tensorflow:

    Args:
        - prev is the activated output of the previous layer
        - n is the number of nodes in the layer to be created
        - activation is the activation function that should be used
            on the output of the layer
        - you should use the tf.layers.Dense layer as the base layer
        with kernal initializer
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        - your layer should incorporate two trainable parameters,
        gamma and beta, initialized as vectors of 1 and 0 respectively
        - you should use an epsilon of 1e-8

    Returns:
        - the activated output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = model(prev)
    mean, variance = tf.nn.moments(Z, axes=[0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-8)
    return activation(Z_norm)
