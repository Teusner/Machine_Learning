import numpy as np


def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """

    mu, sigma = np.mean(X, axis=0), np.std(X, axis=0)
    X_norm = (X - np.ones(X.shape) * mu) / sigma
    return X_norm, mu, sigma
