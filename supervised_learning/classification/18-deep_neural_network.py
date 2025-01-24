#!/usr/bin/env python3
"""
    A class DeepNeuralNetwork that defines a deep neural
    network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    A class DeepNeuralNetwork
    """

    def __init__(self, nx, layers):
        ''' DeepNeuralNetwork class constructor'''
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.nx = nx
        self.layers = layers

        # Initialize weights and biases and validate layers in one loop
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights["W1"] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx))
            else:
                self.__weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]
                ) * np.sqrt(2 / layers[i - 1])
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    # create the getter functions of the deep network
    @property
    def L(self):
        ''' return the L attribute'''
        return self.__L

    @property
    def cache(self):
        ''' return the cache attribute'''
        return self.__cache

    @property
    def weights(self):
        ''' return the weights attribute'''
        return self.__weights

    def forward_prop(self, X):
        '''
            Calculates the forward propagation of
            the deep neural network
        '''
        self.__cache["A0"] = X
        for i in range(self.__L):
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            A = self.__cache["A{}".format(i)]
            Z = np.matmul(W, A) + b
            self.__cache["A{}".format(i + 1)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(self.__L)], self.__cache
