#!/usr/bin/env python3
'''
    function
    def create_layer(prev, n, activation):
'''


import tensorflow as tf


def create_layer(prev, n, activation):
    '''
        function
        def create_layer(prev, n, activation):
    '''
    init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init)
    return layer(prev)
