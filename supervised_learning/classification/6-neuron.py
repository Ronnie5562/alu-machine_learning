#!/usr/bin/env python3
'''
    A class Neuron that defines a single neuron performing
    binary classification:
'''


import numpy as np


class Neuron:
    '''
        Class Neuron
    '''
    def __init__(self, nx):
        '''
            Constructor
        '''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''
            Getter
        '''
        return self.__W

    @property
    def b(self):
        '''
            Getter
        '''
        return self.__b

    @property
    def A(self):
        '''
            Getter
        '''
        return self.__A

    def forward_prop(self, X):
        '''
            Calculates the forward propagation of the neuron
        '''
        self.__A = 1 / (1 + np.exp(-np.dot(self.__W, X) - self.__b))
        return self.__A

    def cost(self, Y, A):
        '''
            Calculates the cost of the model using logistic regression
        '''
        m = Y.shape[1]
        cost = ((-1 / m) * np.sum(Y * np.log(A) + (1 - Y)
                                  * np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        '''
            Evaluates the neuron’s predictions
        '''
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''
            Calculates one pass of gradient descent on the neuron
        '''
        m = Y.shape[1]
        dz = A - Y
        db = (1 / m) * np.sum(dz)
        dw = (1 / m) * np.matmul(X, dz.T)
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)
        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''
            Trains the neuron
        '''
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')

        if iterations < 1:
            raise ValueError('iterations must be a positive integer')

        if type(alpha) is not float:
            raise TypeError('alpha must be a float')

        if alpha < 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            # Forward propagation
            A = self.forward_prop(X)

            # Gradient descent
            self.gradient_descent(X, Y, A, alpha)

        # Evaluation after training
        evaluation = self.evaluate(X, Y)

        return evaluation
