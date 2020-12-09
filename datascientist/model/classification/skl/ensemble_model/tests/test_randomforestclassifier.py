from datascientist.model.classification.skl.ensemble_model.randomforestclassifier import _randomforestclassifier

import numpy as np


def test_randomforestclassifier():
    x_train = np.reshape([100, 120, 160, 180], (-1,1))
    y_train = np.array([0, 0, 1, 1])

    x_test = np.reshape([90, 200, 60, 70], (-1,1))
    y_test = np.array([0, 1, 0, 0])

    metrics = 'f1_score'
    answer = _randomforestclassifier(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'RandomForestClassifier'
    assert round(answer[1] , 1) == 1.0
    assert answer[2] is None

    metrics = 'jaccard_score'
    answer = _randomforestclassifier(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'RandomForestClassifier'
    assert round(answer[1] , 1) == 1.0
    assert answer[2] is None

    metrics = 'accuracy_score'
    answer = _randomforestclassifier(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    assert answer[0] == 'RandomForestClassifier'
    assert round(answer[1] , 1) == 1.0
    assert answer[2] is None

    answer = _randomforestclassifier(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    arr = np.array([0, 1, 0, 0])
    for i in range(len(answer[2])):
        assert round(answer[2][i], 2) == round(arr[i], 2)
