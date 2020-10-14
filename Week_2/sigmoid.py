import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""
    g = 1 / (1 + np.exp(-z))
    return  g
    
if __name__ == "__main__":
    print(sigmoid(-100), sigmoid(0), sigmoid(100))
    a = np.array([[0], [1], [2]])
    print(sigmoid(a))