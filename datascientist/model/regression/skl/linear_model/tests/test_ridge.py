from datascientist.model.regression.skl.linear_model.ridge import _ridge

import numpy as np


def test_ridge():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _ridge(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Ridge'
    assert round(answer[1], 2) == 0.40
    assert answer[2] is None

    metrics = 'mse'
    answer = _ridge(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Ridge'
    assert round(answer[1], 2) == 0.25
    assert answer[2] is None

    metrics = 'rmse'
    answer = _ridge(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Ridge'
    assert round(answer[1], 2) == 0.50
    assert answer[2] is None

    answer = _ridge(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    assert np.any(answer[2] == np.array([6.7,  8.1,  8.9, 10.3]))
