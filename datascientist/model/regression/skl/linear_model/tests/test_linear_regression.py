from datascientist.model.regression.skl.linear_model.linear_regression import _linear_regression

import numpy as np

def test_linear_regression():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _linear_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Linear Regression'
    assert round(answer[1] * 10**16, 2) == 2.22
    assert answer[2] == None

    metrics = 'mse'
    answer = _linear_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Linear Regression'
    assert round(answer[1] * 10**31, 2) == 1.97
    assert answer[2] == None

    metrics = 'rmse'
    answer = _linear_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Linear Regression'
    assert round(answer[1] * 10 **16, 2) == 4.44
    assert answer[2] == None

    answer = _linear_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    assert np.any(answer[2] == np.array([6., 8., 9., 11.]))
