from datascientist.model.regression.skl.linear_model.lassolarscv import _lassolarscv

import numpy as np
from pytest import raises


def test_lassolarscv():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3,4], [4,5]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _lassolarscv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoLarsCV'
    assert round(answer[1] * 10**16, 3) == 4.441
    assert answer[2] is None

    metrics = 'mse'
    answer = _lassolarscv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoLarsCV'
    assert round(answer[1] * 10**31, 3) == 7.889
    assert answer[2] is None

    metrics = 'rmse'
    answer = _lassolarscv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'LassoLarsCV'
    assert round(answer[1] * 10**16, 3) == 8.882
    assert answer[2] is None

    answer = _lassolarscv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr = np.array([ 5.99999773,  8.00000302,  8.99999645, 11.00000174])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)
    
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3
    raises(ValueError, lambda: _lassolarscv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test))
