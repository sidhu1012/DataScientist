from datascientist.model.regression.evaluation_metrics.mse import _mse
from datascientist.model.regression.evaluation_metrics.mae import _mae
from datascientist.model.regression.evaluation_metrics.rmse import _rmse

from sklearn.ensemble import RandomForestRegressor
import numpy as np


def _random_forest_regression(*, train, test, x_predict=None, metrics, n_estimators=100, max_depth=None,
                              min_samples_split=2,
                              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                              max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False,
                              n_jobs=None,
                              random_state=None, verbose=0, warm_start=False):
    """For for info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf)
    model.fit(train[0], train[1])
    model_name = 'RandomForest'
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
