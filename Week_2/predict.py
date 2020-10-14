import numpy as np
from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """
    m = X.shape[0]
    x = sigmoid(X @ theta)
    p = (x > 0.5).astype(int)
    return p
