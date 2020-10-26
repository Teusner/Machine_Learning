import numpy as np
from sigmoid import sigmoid

def sigmoidGradient(z):
    """computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element."""

    g = sigmoid(z) * (np.ones((sigmoid(z)).shape) - sigmoid(z))
    return g
