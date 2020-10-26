from datascientist.model.regression.skl.linear_model.omp import _omp

import numpy as np
from pytest import raises


def test_omp():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _omp(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'OrthogonalMatchingPursuit'
    assert round(answer[1], 2) == 0.25
    assert answer[2] is None

    metrics = 'mse'
    answer = _omp(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'OrthogonalMatchingPursuit'
    assert answer[1] == 0.125
    assert answer[2] is None

    metrics = 'rmse'
    answer = _omp(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'OrthogonalMatchingPursuit'
    assert round(answer[1], 2) == 0.35
    assert answer[2] is None

    answer = _omp(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    assert np.any(answer[2] == np.array([ 6. ,  8.5,  8.5, 11. ]))
