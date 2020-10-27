from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import OrthogonalMatchingPursuitCV
import numpy as np


def _ompcv(*, train, test, x_predict=None, metrics, copy=True, fit_intercept=True,
    normalize=True, max_iter=None, cv=None, n_jobs=None, verbose=False):
    """For more info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#sklearn.linear_model.OrthogonalMatchingPursuitCV
    """

    model = OrthogonalMatchingPursuitCV(fit_intercept=fit_intercept, copy=copy, normalize=normalize,
    max_iter=max_iter, cv=cv, n_jobs=n_jobs, verbose=verbose)
    model.fit(train[0], train[1])
    model_name = 'OrthogonalMatchingPursuitCV'
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
