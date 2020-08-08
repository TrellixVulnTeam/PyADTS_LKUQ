import numpy as np

import lightgbm as lgb


class LightGBM(object):
    """
    LightGBM for KPI anomaly detection
    """
    def __init__(self, epochs=20, num_leaves=31, objective='binary', metric='binary_logloss'):
        self.param = {}
        self.param['num_leaves'] = num_leaves
        self.param['objective'] = objective
        self.param['metric'] = metric
        
        self.epochs = 20
        self.bst = None

    def fit(self, x, y):
        train_data = lgb.Dataset(x, label=y)
        self.bst = lgb.train(self.param, train_data, self.epochs)

    def predict(self, x):
        assert(self.bst is not None)
        y_pred = self.bst.predict(x)

        return y_pred

    def reset_param(self, num_leaves=31, objective='binary', metric='binary_logloss'):
        self.param['num_leaves'] = num_leaves
        self.param['objective'] = objective
        self.param['metric'] = metric
