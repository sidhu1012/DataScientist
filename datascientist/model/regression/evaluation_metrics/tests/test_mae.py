import numpy as np
from datascientist.model.regression.evaluation_metrics.mae import _mae

def test_mse():
    a = np.array([0, 1, 2])
    b = np.array([1, 2, 3])
    
    assert _mae(a, b) == 1.0