from datascientist.model.regression.skl.linear_model.lassocv import _lassocv

import numpy as np
from pytest import raises


def test_lassocv():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3,4], [4,5]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _lassocv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoCV'
    assert round(answer[1], 6) == 0.003585
    assert answer[2] is None

    metrics = 'mse'
    answer = _lassocv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoCV'
    assert round(answer[1] * 10**5, 4) == 1.7340
    assert answer[2] is None

    metrics = 'rmse'
    answer = _lassocv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoCV'
    assert round(answer[1], 5) == 0.00416
    assert answer[2] is None

    answer = _lassocv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr = np.array([6.0030983,  8.00717093,  8.9981154, 11.00218803])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)

x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3
    raises(ValueError, lambda: _lassocv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test))
