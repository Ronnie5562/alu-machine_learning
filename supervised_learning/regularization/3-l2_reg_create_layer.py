#!/usr/bin/env python3
'''
    a function def l2_reg_create_layer(prev, n, activation, lambtha):
    that creates a tensorflow layer that includes L2 regularization:
'''


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''
        Function to create a tensorflow layer that includes L2 regularization

        Args:
            - prev: tensor containing the output of the previous layer
            - n: number of nodes the new layer should contain
            - activation: activation function that should be used on the layer
            - lambtha: L2 regularization parameter

        Returns:
            - the output of the new layer
    '''
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=kernel,
                            kernel_regularizer=l2)
    return layer(prev)
