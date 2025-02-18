#!/usr/bin/env python3
"""
    A function that calculates
    the accuracy of a prediction:
"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    that calculates the accuracy of a prediction:
    """
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1)), tf.float32)
    )
