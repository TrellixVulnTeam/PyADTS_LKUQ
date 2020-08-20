import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..base import BaseModel


class RandomForest(BaseModel):
    def __init__(self, n_estimators=100, max_depth=10, num_thread=4):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=num_thread)

    def fit(self, X: np.ndarray, y: np.ndarray=None):
        self.__store_train_data(X, y)

        self.model.fit(X, y)

    def predict_score(self, X: np.ndarray):
        self.__check_fitted()

        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray):
        assert self.__check_fitted()

        return self.model.predict(X)

    def predict_prob(self, X: np.ndarray):
        assert self.__check_fitted()

        return self.model.predict_proba(X)
