#!/usr/bin/env python3
'''
    That returns two placeholders,
    x and y, for the neural network
'''


import tensorflow as tf


def create_placeholders(nx, classes):
    '''
        that returns two placeholders,
        x and y, for the neural network
    '''
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
