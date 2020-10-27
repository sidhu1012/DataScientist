from datascientist.model.regression.skl.linear_model.ard import _ard

import numpy as np
from pytest import raises


def test_ard():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _ard(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ARDRegression'
    assert round(answer[1] * 10**7, 2) == 5.00
    assert answer[2] is None

    metrics = 'mse'
    answer = _ard(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ARDRegression'
    assert round(answer[1] * 10**13, 2) == 3.12
    assert answer[2] is None

    metrics = 'rmse'
    answer = _ard(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ARDRegression'
    assert round(answer[1] * 10**7, 2) == 5.59
    assert answer[2] is None

    answer = _ard(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr = np.array([ 6.00000025,  8.00000075,  8.99999925, 10.99999975])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)
