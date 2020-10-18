"""Root Mean Squared Error."""

import numpy as np

def _rmse(y , y_hat):
    """
    Parameters
    -----------

    y : numpy array
    Actual value.

    y_hat : numpy array
    Predicted value.

    """

    rmse = np.sqrt(np.mean((y - y_hat) ** 2))
    return rmse
    