from datascientist.model.regression.skl.linear_model.lassolars import _lassolars

import numpy as np


def test_lassolars():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _lassolars(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoLars'
    assert answer[1] == 1.5
    assert answer[2] is None

    metrics = 'mse'
    answer = _lassolars(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoLars'
    assert answer[1] == 3.25
    assert answer[2] is None

    metrics = 'rmse'
    answer = _lassolars(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoLars'
    assert round(answer[1], 2) == 1.80
    assert answer[2] is None

    answer = _lassolars(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    assert np.any(answer[2] == np.array([8.5, 8.5, 8.5, 8.5]))
test_lassolars()