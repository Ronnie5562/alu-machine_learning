#!/usr/bin/env python3
"""
    Function def moving_average(data, beta):
    that calculates the weighted moving average of a data set:
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Args:
        - data is the list of data to calculate the moving average of
        - beta is the weight used for the moving average
        - Your moving average calculation should use bias correction
    """
    V = 0
    moving_avg = []
    for i in range(len(data)):
        V = beta * V + (1 - beta) * data[i]
        moving_avg.append(V / (1 - beta ** (i + 1)))
    return moving_avg
