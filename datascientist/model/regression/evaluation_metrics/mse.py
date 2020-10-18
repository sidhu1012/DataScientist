"""Mean Square Error."""

import numpy as np

def _mse(y, y_hat):
    """
    Parameters
    -----------

    y : numpy array
    Actual value.

    y_hat : numpy array
    Predicted value.

    """

    mse = np.mean((y_hat - y) ** 2)
    return mse
