import numpy as np

def computeCostMulti(X, y, theta):  
	"""
	   computes the cost of using theta as the parameter for linear 
	   regression to fit the data points in X and y
	"""
	m = y.size
	J = 0.
	for i in range(m):
		J += 1/(2*m) * (X[i]@theta - y[i])**2
	return J
