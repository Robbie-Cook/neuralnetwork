"""
Model network
This is a network based off neupy (neupy.com)
"""

from neupy import layers
import theano.tensor as T

network = layers.join(
    layers.Input(32),
    layers.Sigmoid(16),
    layers.Sigmoid(32),
)

predict = network.compile()
print(T.matrix())
