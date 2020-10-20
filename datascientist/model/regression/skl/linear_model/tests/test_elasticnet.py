from datascientist.model.regression.skl.linear_model.elasticnet import _elasticnet

import numpy as np


def test_elasticnet():
    x_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(x_train, np.array([1, 2])) + 3

    x_test = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_test = np.dot(x_test, np.array([1, 2])) + 3

    metrics = 'mae'
    answer = _elasticnet(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ElasticNet'
    assert round(answer[1], 3) == 1.091
    assert answer[2] is None

    metrics = 'mse'
    answer = _elasticnet(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ElasticNet'
    assert round(answer[1], 3) == 1.595
    assert answer[2] is None

    metrics = 'rmse'
    answer = _elasticnet(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'ElasticNet'
    assert round(answer[1], 3) == 1.263
    assert answer[2] is None

    answer = _elasticnet(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr = np.array([7.72727248, 8.45454497, 8.54545503, 9.27272752])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)
