"""Mean Absolute Error."""

import numpy as np

def _mae(y, y_hat):
    """
    Parameters
    -----------

    y : numpy array
    Actual value.

    y_hat : numpy array
    Predicted value.

    """

    mae = np.mean(np.absolute(y_hat - y))
    return mae
