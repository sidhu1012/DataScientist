from datascientist.model.regression.skl.linear_model.lars import _lars

import numpy as np


def test_lars():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _lars(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Lars'
    assert round(answer[1] * 10**16, 2) == 2.22 
    assert answer[2] is None

    metrics = 'mse'
    answer = _lars(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Lars'
    assert round(answer[1] * 10**31, 3) == 1.972
    assert answer[2] is None

    metrics = 'rmse'
    answer = _lars(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Lars'
    assert round(answer[1] * 10**16, 3) == 4.441
    assert answer[2] is None

    answer = _lars(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    assert np.any(answer[2] == np.array([6., 8., 9., 11.]))
