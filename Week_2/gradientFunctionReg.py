import numpy as np
from sigmoid import sigmoid


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    
    # Initialize some useful values
    m,n = X.shape   # number of training examples and parameters
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    # =============================================================
    l = Lambda * np.vstack((np.array([[0]]), np.ones((n-1, 1))))
    inner = ((sigmoid(X @ theta) - y).T @ X).T
    grad = (np.sum(inner, axis=1).reshape(n, 1) + l * theta)/m

    # =============================================================

    return grad