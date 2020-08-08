import numpy as np

from sklearn.ensemble import RandomForestClassifier


class RandomForest(object):
    """

    """
    def __init__(self, n_estimators=100, max_depth=10, num_thread=14):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=num_thread)
        self.is_trained = False

    def fit(self, x, y):
        self.model.fit(x, y)
        self.is_trained = True
    
    def predict(self, x):
        assert(self.is_trained)
        y_pred = self.model.predict(x)

        return y_pred
