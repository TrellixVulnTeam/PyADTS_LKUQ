import numpy as np

from pyadts.model.base import BaseModel


class ThresholdDetector(BaseModel):
    def __init__(self, high: float, low: float):
        super(ThresholdDetector, self).__init__()

        self.__train_x = None
        self.__train_y = None

        self.__low = low
        self.__high = high

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        # Do nothing
        pass

    def predict_score(self, x: np.ndarray):
        return np.abs(x - (self.__low + self.__high) / 2)

    def predict(self, x: np.ndarray):
        assert self.check_fitted()

        predictions = np.zeros(x.shape[0])
        predictions[x >= self.__high] = 1
        predictions[x <= self.__low] = 1

        return predictions.astype(np.int)
