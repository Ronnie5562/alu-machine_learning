#!/usr/bin/env python3
"""
    Function def early_stopping(cost, opt_cost, threshold, patience, count):
    that determines if you should stop gradient descent early:
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early

    Args:
        - cost: current validation cost of the neural network
        - opt_cost: lowest recorded validation cost of the neural network
        - threshold: threshold used for early stopping
        - patience: patience count used for early stopping
        - count: how long the threshold has not been met

    Returns:
        - Returns: a boolean of whether the network should be stopped early
        followed by the updated count
    """
    if cost < opt_cost - threshold:
        opt_cost = cost
        count = 0
        return (False, count)
    if cost >= opt_cost - threshold:
        count += 1
        if count == patience:
            return (True, count)
        return (False, count)
