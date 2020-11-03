import numpy as np


def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m,n = X.shape # number of training examples
    theta = theta.reshape((n,1)) # in case where theta is a vector (n,) 

    J = 1/(2*m) * (np.sum((X @ theta - y)**2) + Lambda * np.sum(theta**2))

    lin = np.vstack((np.zeros((1, 1)), np.ones((n-1, 1))))
    inner = ((X @ theta - y).T @ X)
    grad = 1/m * (inner + Lambda * lin * theta.T)

    return J.flatten(), grad.flatten()