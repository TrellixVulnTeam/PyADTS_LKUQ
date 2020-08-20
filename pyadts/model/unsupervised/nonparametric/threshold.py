import numpy as np

from ...base import BaseModel


class ThresholdDetector(BaseModel):
    def __init__(self, high: float, low: float):
        super(ThresholdDetector, self).__init__()

        self.__train_x = None
        self.__train_y = None

        self.__low = low
        self.__high = high

    def fit(self, X: np.ndarray, y: np.ndarray=None):
        # Do nothing
        pass

    def predict_score(self, X: np.ndarray):
        return np.abs(X - (self.__low + self.__high)/2)

    def predict(self, X: np.ndarray):
        assert self.__check_fitted()

        predictions = np.zeros(X.shape[0])
        predictions[X >= self.__high] = 1
        predictions[X <= self.__low] = 1

        return predictions.astype(np.int)
