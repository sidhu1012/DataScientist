from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


def _gradient_boosting_regression(*, train, test, x_predict=None, metrics,
                                  loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                  min_samples_split=2, min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                  min_impurity_split=None, init=None, random_state=None, max_features=None,
                                  alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated',
                                  validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
    """For more info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    """
    model = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                      subsample=subsample,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth,
                                      min_impurity_decrease=min_impurity_decrease,
                                      min_impurity_split=min_impurity_split,
                                      init=init, random_state=random_state, max_features=max_features,
                                      alpha=alpha, verbose=verbose, max_leaf_nodes=max_leaf_nodes,
                                      warm_start=warm_start,
                                      presort=presort, validation_fraction=validation_fraction,
                                      n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

    model.fit(train[0], train[1])
    model_name = 'GradientBoostingRegressor'
    y_hat = model.predict(test[0])

    if metrics == 'mse':
        accuracy = _mse(test[1], y_hat)
    if metrics == 'rmse':
        accuracy = _rmse(test[1], y_hat)
    if metrics == 'mae':
        accuracy = _mae(test[1], y_hat)

    if x_predict is None:
        return model_name, accuracy, None

    y_predict = model.predict(x_predict)
    return model_name, accuracy, y_predict
