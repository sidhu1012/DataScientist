from datascientist.model.regression.skl.linear_model.ompcv import _ompcv

import numpy as np
from pytest import raises


def test_ompcv():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3,4], [4,5]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _ompcv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    print(answer)
    assert answer[0] == 'OrthogonalMatchingPursuitCV'
    assert round(answer[1] * 10**15, 2) == 1.78
    assert answer[2] is None

    metrics = 'mse'
    answer = _ompcv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    print(answer)
    assert answer[0] == 'OrthogonalMatchingPursuitCV'
    assert round(answer[1] * 10**30, 2) == 4.73
    assert answer[2] is None

    metrics = 'rmse'
    answer = _ompcv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    print(answer)
    assert answer[0] == 'OrthogonalMatchingPursuitCV'
    assert round(answer[1] * 10**6, 2) == 3.00
    assert answer[2] is None

    answer = _ompcv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    print(answer)
    arr = np.array([ 5.99999765,  8.00000345,  8.99999615, 11.00000194])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)

    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3
    raises(ValueError, lambda: _ompcv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test))
