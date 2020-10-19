from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import RidgeCV
import numpy as np


def _ridgecv(*, train, test, x_predict=None, metrics, alphas=(0.1, 1.0, 10.0), 
    fit_intercept=True, normalize=False, scoring=None, cv=None, gcv_mode=None, store_cv_values=False):
    """For more info visit:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV
    """
    model = RidgeCV(alphas=alphas, fit_intercept=fit_intercept, normalize=normalize, scoring=scoring,
        cv=cv, gcv_mode=gcv_mode, store_cv_values=store_cv_values)
    model.fit(train[0], train[1])
    model_name = 'RidgeCV'
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
