import abc

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from pyadt.model.utils.__auto_threshold import auto_threshold

cla


class BaseModel(abc.ABC):
    def __init__(self):
        pass

    def __store_train_data(self, X:np.ndarray, y: np.ndarray=None):
        self.__train_x = X
        self.__train_y = y

    def __check_fitted(self):
        return hasattr(self, '__train_x')

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray=None):
        pass

    @abc.abstractmethod
    def predict_score(self, X: np.ndarray):
        pass

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_params(self):
        return self.__dict__

    def predict(self, X: np.ndarray):
        assert self.__check_fitted()

        train_score = self.predict_score(self.__train_x)
        test_score = self.predict_score(X)

        th = auto_threshold(train_score, test_score)

        predictions = np.zeros_like(test_score)
        predictions[test_score < th] = 1

        return predictions.astype(np.int)

    def predict_prob(self, X: np.ndarray):
        assert self.__check_fitted()

        train_score = self.predict_score(self.__train_x)
        test_score = self.predict_score(X)

        scaler = MinMaxScaler().fit(train_score.reshape(-1, 1))
        prob = scaler.transform(test_score.reshape(-1, 1)).ravel().clip(0, 1).reshape(-1)

        return prob

    def fit_predict(self, X: np.ndarray, y: np.ndarray=None):
        self.fit(X, y)

        return self.predict(X)

    def fit_predict_score(self, X: np.ndarray, y: np.ndarray = None):
        self.fit(X, y)

        return self.predict_score(X)

    def fit_predict_prob(self, X: np.ndarray, y: np.ndarray = None):
        self.fit(X, y)

        return self.predict_prob(X)
