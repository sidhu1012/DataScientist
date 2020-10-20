from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import ElasticNet
import numpy as np


def _elasticnet(*, train, test, x_predict=None, metrics, alpha=1.0, l1_ratio=0.5,
    fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True,
    tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
    """For more info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet"""

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, normalize=normalize,
        precompute=precompute, max_iter=max_iter, copy_X=copy_X, tol=tol, warm_start=warm_start,
        positive=positive, random_state=random_state, selection=selection)
    model.fit(train[0], train[1])
    model_name = 'ElasticNet'
    y_hat = model.predict(test[0])

    if metrics == 'mse':
        accuracy = _mse(test[1], y_hat)
    if metrics == 'rmse':
        accuracy = _rmse(test[1], y_hat)
    if metrics == 'mae':
        accuracy = _mae(test[1], y_hat)

    if x_predict is None:
        return (model_name, accuracy, None)

    y_predict = model.predict(x_predict)
    return (model_name, accuracy, y_predict)
