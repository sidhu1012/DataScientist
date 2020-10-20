from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import LassoLars
import numpy as np


def _lassolars(*, train, test, x_predict=None, metrics, alpha=1.0, fit_intercept=True,
    verbose=False, normalize=True, precompute='auto', max_iter=500, eps=2.220446049250313e-16,
    copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None):
    """For more info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars
    """

    model = LassoLars(alpha=alpha, fit_intercept=fit_intercept, verbose=verbose, normalize=normalize,
        precompute=precompute, max_iter=max_iter, eps=eps, copy_X=copy_X, fit_path=fit_path, 
        positive=positive, jitter=jitter, random_state=random_state)
    model.fit(train[0], train[1])
    model_name = 'LassoLars'
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
