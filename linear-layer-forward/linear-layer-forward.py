import numpy as np

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Convert all input type list to numpy array type, so this can calculate dot product in numpy 
    X = np.asarray(X)
    W = np.asarray(W)
    b = np.asarray(b)

    # Caculate dot product between X and W, and plub bias b, then convert that result to list type follow requirements by using tolist() method
    return (np.dot(X, W) + b).tolist()