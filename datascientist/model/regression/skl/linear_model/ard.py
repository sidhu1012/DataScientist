from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import ARDRegression
import numpy as np


def _ard(*, train, test, x_predict=None, metrics, n_iter=300, tol=0.001, alpha_1=1e-06, 
    alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000.0, 
    fit_intercept=True, normalize=False, copy_X=True, verbose=False):
    """For more info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression
    """

    model = ARDRegression(n_iter=n_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,
        lambda_2=lambda_2, compute_score=compute_score, threshold_lambda=threshold_lambda, 
        fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, verbose=verbose)
    model.fit(train[0], train[1])
    model_name = 'ARDRegression'
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
