#!/usr/bin/env python3
'''
    a class Exponential that represents
    an exponential distribution
'''


class Exponential:
    '''
        Class Exponential that represents
        an exponential distribution
    '''

    def __init__(self, data=None, lambtha=1.):
        '''
            Class constructor
        '''
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        '''
            Calculates the value of the
            PDF for a given time period
        '''
        if x < 0:
            return 0
        return (self.lambtha * (2.7182818285 ** (-self.lambtha * x)))

    def cdf(self, x):
        '''
            Calculates the value of the
            CDF for a given time period
        '''
        if x < 0:
            return 0
        return 1 - (2.7182818285 ** (-self.lambtha * x))
