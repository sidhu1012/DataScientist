import numpy as np
import datascientist.model.regression.evaluation_metrics.rmse import _rmse

def test_mse():
    a = np.array([1, 2, 3])
    b = np.array([0, 1, 1])
    
    assert round(_rmse(a, b), 2) == 1.41