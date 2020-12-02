from datascientist.model.regression.skl.ensemble_model.gradient_boosting_regression import _gradient_boosting_regression

import numpy as np


def test_gradient_boosting():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _gradient_boosting_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'GradientBoostingRegressor'
    assert round(answer[1] * 10**4, 2) == 0.4
    assert answer[2] is None

    metrics = 'mse'
    answer = _gradient_boosting_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'GradientBoostingRegressor'
    assert round(answer[1] * 10**8, 2) == 9.10
    assert answer[2] is None

    metrics = 'rmse'
    answer = _gradient_boosting_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'GradientBoostingRegressor'
    assert round(answer[1] * 10 ** 4, 2) == 3.02
    assert answer[2] is None

    answer = _gradient_boosting_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics,
                                           x_predict=x_test)
    arr = np.array([6.00043558,  8.00005297,  8.99991706, 10.99959438])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)
