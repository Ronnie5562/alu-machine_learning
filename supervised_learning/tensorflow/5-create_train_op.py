#!/usr/bin/env python3
'''
    Train the model
'''

import tensorflow as tf


def create_train_op(loss, alpha):
    '''
        that creates the training
        operation for the network:
    '''
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
