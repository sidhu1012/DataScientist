from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import Ridge
import numpy as np

def _ridge(*, train, test, x_predict=None, metrics, alpha=1.0, fit_intercept=True, 
    normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
    """For for info visit : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge"""

    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
        max_iter=max_iter, tol=tol, solver=solver, random_state=random_state)
    model.fit(train[0], train[1])
    model_name = 'Ridge'
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