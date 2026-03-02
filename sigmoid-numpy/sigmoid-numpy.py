import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # - Convert input list to numpy array type, so it can caculate in each elements in numpy array
    x = np.asarray(x, dtype=float)

    # - This formular will apply for each elements in numpy array and return as an numpy array 
    return 1 / (1 + np.exp(-x))