#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


Neuron = __import__('7-neuron').Neuron

lib_train = np.load('../../../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

np.random.seed(1)
neuron = Neuron(X_train.shape[0])
print(neuron.train(X_train, Y_train, iterations=100, verbose=False, graph=False))
