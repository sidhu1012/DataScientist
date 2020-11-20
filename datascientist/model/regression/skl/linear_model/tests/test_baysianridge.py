from datascientist.model.regression.skl.linear_model.baysianridge import _bayesianridge

import numpy as np


def test_baysianridge():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _bayesianridge(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Bayesian Ridge'
    assert round(answer[1] * 10**7, 5) == 2.0
    assert answer[2] is None

    metrics = 'mse'
    answer = _bayesianridge(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Bayesian Ridge'
    assert round(answer[1] * 10**14, 4) == 8.0
    assert answer[2] is None

    metrics = 'rmse'
    answer = _bayesianridge(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'Bayesian Ridge'
    assert round(answer[1] * 10**7, 4) == 2.8284
    assert answer[2] is None

    answer = _bayesianridge(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr = np.array([ 6.0000004,  8.       ,  9.       , 10.9999996])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)
        