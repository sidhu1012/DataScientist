from datascientist.model.regression.skl.linear_model.ridgecv import _ridgecv

import numpy as np


def test_ridgecv():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _ridgecv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'RidgeCV'
    assert round(answer[1], 4) == 0.0496
    assert answer[2] is None

    metrics = 'mse'
    answer = _ridgecv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'RidgeCV'
    assert round(answer[1], 4) == 0.0046
    assert answer[2] is None

    metrics = 'rmse'
    answer = _ridgecv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'RidgeCV'
    assert round(answer[1], 4) == 0.0675
    assert answer[2] is None

    answer = _ridgecv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr = np.array([6.09541985,  8.00381679,  8.99618321, 10.90458015])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)
