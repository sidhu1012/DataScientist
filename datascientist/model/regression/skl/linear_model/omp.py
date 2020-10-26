from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np


def _omp(*, train, test, x_predict=None, metrics,  n_nonzero_coefs=None, tol=None, 
    fit_intercept=True, normalize=True, precompute='auto'):
    """For more info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit
    """

    model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, tol=tol, fit_intercept=fit_intercept,
        normalize=normalize, precompute=precompute)
    model.fit(train[0], train[1])
    model_name = 'OrthogonalMatchingPursuit'
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
