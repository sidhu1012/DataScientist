from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import LassoCV
import numpy as np


def _lassocv(*, train, test, x_predict=None, metrics, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True,
        normalize=False, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=None, verbose=False,
        n_jobs=None, positive=False, random_state=None, selection='cyclic'):
    """For for info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV
    """
    model = LassoCV(eps=eps, n_alphas=n_alphas, alphas=alphas, fit_intercept=fit_intercept, normalize=normalize,
        precompute=precompute, max_iter=max_iter, tol=tol, copy_X=copy_X, cv=cv, verbose=verbose, n_jobs=n_jobs,
        positive=positive, random_state=random_state, selection=selection)
    model.fit(train[0], train[1])
    model_name = 'LassoCV'
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