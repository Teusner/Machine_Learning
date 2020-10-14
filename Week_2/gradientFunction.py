from sigmoid import sigmoid
import numpy as np


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic 
    regression and the gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    # number of training examples 
    m = X.shape[0]   

    # number of parameters
    n = X.shape[1]   
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc

    # gradient variable
    grad = 0.
    for i in range(m):
        grad += (sigmoid(X[i].reshape(1, n) @ theta) - y[i]) @ X[i].reshape(1, n)
    grad *= 1/m
    return grad