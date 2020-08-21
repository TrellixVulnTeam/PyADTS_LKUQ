import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..base import BaseModel


class RandomForest(BaseModel):
    def __init__(self, n_estimators=100, max_depth=10, num_thread=4):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=num_thread)

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        self.store_train_data(x, y)

        self.model.fit(x, y)

    def predict_score(self, x: np.ndarray):
        self.check_fitted()

        return self.model.predict_proba(x)

    def predict(self, x: np.ndarray):
        assert self.check_fitted()

        return self.model.predict(x)

    def predict_prob(self, x: np.ndarray):
        assert self.check_fitted()

        return self.model.predict_proba(x)
