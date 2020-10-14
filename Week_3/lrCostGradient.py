import numpy as np
from sigmoid import sigmoid

def lrCostGradient(theta, X, y, Lambda):
    """computes the gradient of the cost  w.r.t. to the parameters 
    theta for regularized logistic regression .
    """

    # préambule
    m,n = X.shape # m = 5; n = 4
    theta = theta.reshape((n,1)) # (4,1)

    lin = np.vstack((np.zeros((1, 1)), np.ones((n-1, 1))))
    inner = (sigmoid(X @ theta) - y).T @ X
    grad = 1/m * (np.sum(inner, axis=0) + Lambda * lin * theta)
    return grad[0].flatten()
