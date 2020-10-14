import numpy as np
from sigmoid import sigmoid

def lrCostFunction(theta, X, y, Lambda):
    """computes the cost of using
    theta as the parameter for regularized logistic regression.
    """

    # preambule
    m,n = X.shape # 5,4
    theta = theta.reshape((n,1)) # (4,1)

    inner = - y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta))
    J = 1/m * (np.sum(inner) + Lambda/2 * np.sum(theta**2))
    return J
