import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression."""

	# Initialize some useful values
    m,n = X.shape   # number of training examples and parameters
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc

    J = 0.
    for i in range(m):
        J += - y[i] * np.log(sigmoid(X[i].T @ theta)) - (1 - y[i]) * np.log(1 - sigmoid(X[i].T @ theta))
    J *= 1/m
    return J
