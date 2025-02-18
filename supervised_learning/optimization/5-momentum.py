#!/usr/bin/env python3
'''
    Function def update_variables_momentum(alpha, beta1, var, grad, v):
    that updates a variable using the gradient descent with momentum
    optimization algorithm:
'''


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''
        Args:
            - alpha is the learning rate
            - beta1 is the momentum weight
            - var is a numpy.ndarray containing the variable to be updated
            - grad is a numpy.ndarray containing the gradient of var
            - v is the previous first moment of var

        Returns:
            The updated variable and the new moment, respectively
    '''
    V = beta1 * v + (1 - beta1) * grad
    var = var - alpha * V
    return var, V
