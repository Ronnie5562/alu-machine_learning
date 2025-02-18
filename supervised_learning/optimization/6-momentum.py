#!/usr/bin/env python3
'''
    function def create_momentum_op(loss, alpha, beta1):
    that creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm
'''


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    '''
        Creates the training operation for a neural network in tensorflow
        using the gradient descent with momentum optimization algorithm
    '''
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train_op = optimizer.minimize(loss)
    return train_op
