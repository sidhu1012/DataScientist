from datascientist.model.regression.skl.linear_model.lassolarsic import _lassolarsic

import numpy as np
from pytest import raises


def test_lassolarsic():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _lassolarsic(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoLarsIC'
    assert round(answer[1], 2) == 0.85
    assert answer[2] is None

    metrics = 'mse'
    answer = _lassolarsic(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoLarsIC'
    assert round(answer[1], 2) == 0.85
    assert answer[2] is None

    metrics = 'rmse'
    answer = _lassolarsic(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoLarsIC'
    assert round(answer[1], 2) == 0.92
    assert answer[2] is None

    answer = _lassolarsic(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr = np.array([7.20710678, 8.5       , 8.5       , 9.79289322])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)
