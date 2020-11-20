from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import BayesianRidge
import numpy as np


def _bayesianridge(*, train, test, x_predict=None, metrics, n_iter=300, tol=0.001, 
        alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, 
        lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, 
        verbose=False):
    """For more info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
    """

    model = BayesianRidge(n_iter=n_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,
        lambda_2=lambda_2, alpha_init=alpha_init, lambda_init=lambda_init, compute_score=compute_score,
        fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, verbose=verbose)
    model.fit(train[0], train[1])
    model_name = 'Bayesian Ridge'
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
    