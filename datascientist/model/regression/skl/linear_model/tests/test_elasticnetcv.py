from datascientist.model.regression.skl.linear_model.elasticnetcv import _elasticnetcv

import numpy as np


def test_elasticnetcv():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3,4], [4,5]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _elasticnetcv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ElasticNetCV'
    assert round(answer[1], 4) == 0.0088
    assert answer[2] is None

    metrics = 'mse'
    answer = _elasticnetcv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ElasticNetCV'
    assert round(answer[1], 6) == 0.000122
    assert answer[2] is None

    metrics = 'rmse'
    answer = _elasticnetcv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ElasticNetCV'
    assert round(answer[1], 3) == 0.011
    assert answer[2] is None

    answer = _elasticnetcv(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr = np.array([6.01765141,  8.00036361,  9.01240037, 10.99511257])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)
