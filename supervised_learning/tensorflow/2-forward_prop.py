#!/usr/bin/env python3
'''
    Forward propagation for
    def forward_prop(x, layer_sizes, activations)
'''


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes, activations):
    '''
        that calculates the forward propagation
    '''
    output = x
    for size, activation in zip(layer_sizes, activations):
        output = create_layer(output, size, activation)
    return output
