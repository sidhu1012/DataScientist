from datascientist.model.regression.skl.ensemble_model.random_forest_regression import _random_forest_regression

import numpy as np


def test_random_forest():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _random_forest_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Random Forest Regression'
    assert round(answer[1] * 10, 3) == 3.975
    assert answer[2] is None

    metrics = 'mse'
    answer = _random_forest_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Random Forest Regression'
    assert round(answer[1] * 10, 2) == 2.35
    assert answer[2] is None

    metrics = 'rmse'
    answer = _random_forest_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Random Forest Regression'
    assert round(answer[1], 2) == 0.48
    assert answer[2] is None

    answer = _random_forest_regression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics,
                                       x_predict=x_test)
    assert np.any(answer[2] == np.array([6.66,  7.8,  8.95, 10.32]))
