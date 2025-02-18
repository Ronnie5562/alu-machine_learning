#!/usr/bin/env python3
"""
    Function def learning_rate_decay
    (alpha, decay_rate, global_step, decay_step):
    that updates the learning rate using inverse
    time decay in numpy
"""


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy

    Args:
        - alpha is the original learning rate
        - decay_rate is the decay rate
        - global_step is the number of passes of gradient
        descent that have elapsed
        - decay_step is the number of passes of gradient
        descent that should occur
        before the learning rate is decayed
        - the learning rate decay should occur in a stepwise
        fashion

    Returns:
        The updated learning rate
    """
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
