import datascientist.model.regression.evaluation_metrics.mse import _mse
import datascientist.model.regression.evaluation_metrics.mae import _mae
import datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.linear_model import LinearRegression
import numpy as np

def _linear_regression(*, train=(x_train, y_train), test=(x_test, y_test), x_predict=None, metrics, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
    """For more info visit : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"""

    model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)
    model.fit(train[0], train[1])
    model_name = 'Linear Regression'
    y_hat = model.predict(test[0])

    if metrics == 'mse':
        accuracy = _mse(test[1], y_hat)
    if metrics = 'rmse':
        accuracy = _rmse(test[1], y_hat)
    if metrics = 'mae':
        accuracy = _mae(test[1], y_hat)

    if x_predict is None:
        return (model_name, accuracy, None)

    y_predict = model.predict(x_predict)
    return (model_name, accuracy, y_predict)
