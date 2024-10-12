#!/usr/bin/env python3
'''
    Poisson distribution
    that represents a poisson distribution
'''


class Poisson:
    '''
        Class Poisson that represents a
        distribution of Poisson
    '''

    def factorial(self, k):
        '''
            Calculates the factorial
        '''
        if k < 0:
            return 0
        if k == 0 or k == 1:
            return 1
        return k * self.factorial(k - 1)

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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        '''
            Calculates the value of the
            PMF for a given number of successes
        '''
        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285
        return ((self.lambtha ** k) * (e ** (-self.lambtha))
                ) / (self.factorial(k))

    def cdf(self, k):
        '''
            Calculates the value of the
            CDF for a given number of successes
        '''
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
