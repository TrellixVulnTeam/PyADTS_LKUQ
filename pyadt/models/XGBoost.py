import xgboost as xgb


class XGBoost(object):
    """
    XGBoost for KPI anomaly detection
    """

    def __init__(self, epochs=20, max_depth=2, eta=1, objective='binary:logistic', nthread=14):
        self.param = {}
        self.param['max_depth'] = max_depth
        self.param['eta'] = eta
        self.param['objective'] = objective
        self.param['nthread'] = nthread

        self.epochs = epochs
        self.bst = None

    def fit(self, x, y):
        data_train = xgb.DMatrix(x, label=y)
        self.bst = xgb.train(self.param, data_train, self.epochs)

    def predict(self, x):
        assert (self.bst is not None)
        data_test = xgb.DMatrix(x)
        y_pred = self.bst.predict(data_test)

        return y_pred

    def reset_param(self, max_depth=2, eta=1, objective='binary:logistic', nthread=14):
        self.param['max_depth'] = max_depth
        self.param['eta'] = eta
        self.param['objective'] = objective
        self.param['nthread'] = nthread
