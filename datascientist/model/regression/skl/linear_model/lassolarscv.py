from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import LassoLarsCV
import numpy as np


def _lassolarscv(*, train, test, x_predict=None, metrics, fit_intercept=True, verbose=False,
    max_iter=500, normalize=True, precompute='auto', cv=None, max_n_alphas=1000, n_jobs=None, 
    eps=2.220446049250313e-16, copy_X=True, positive=False):
    """For more info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV
    """

    model = LassoLarsCV(fit_intercept=fit_intercept, verbose=verbose, max_iter=max_iter,
        normalize=normalize, precompute=precompute, cv=cv, max_n_alphas=max_n_alphas, n_jobs=n_jobs,
        eps=eps, copy_X=copy_X, positive=positive)
    model.fit(train[0], train[1])
    model_name = 'LassoLarsCV'
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
