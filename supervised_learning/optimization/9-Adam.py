#!/usr/bin/env python3
'''
    Function def create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    that creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm:
'''


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''
        Creates the training operation for a neural network in tensorflow
        using the Adam optimization algorithm

        Args:
            - alpha is the learning rate
            - beta1 is the weight used for the first moment
            - beta2 is the weight used for the second moment
            - epsilon is a small number to avoid division by zero
            - var is a numpy.ndarray containing the variable to be updated
            - grad is a numpy.ndarray containing the gradient of var
            - v is the previous first moment of var
            - s is the previous second moment of var
            - t is the time step used for bias correction
        Returns:
            - the updated variable, the new first moment, and the
            new second moment, respectively
    '''
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * np.power(grad, 2)
    v_corrected = v / (1 - np.power(beta1, t))
    s_corrected = s / (1 - np.power(beta2, t))
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var, v, s
