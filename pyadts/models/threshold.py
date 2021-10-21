import numpy as np

from pyadts.models.base import BaseModel


class ThresholdDetector(BaseModel):
    def __init__(self, high: float, low: float):
        super(ThresholdDetector, self).__init__()

        self.__low = low
        self.__high = high

        if self.__low >= self.__high:
            raise ValueError('The parameter `high` must be larger than `low`!')

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        assert x.ndim==1 or (x.ndim==2 and x.shape[1] == 1), 'Only support 1-dimensional time series!'
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        self.store_train_data(x, y)

    def predict_score(self, x: np.ndarray):
        assert self.check_fitted()
        return np.abs(x - (self.__low + self.__high) / 2).reshape(-1)

    def predict(self, x: np.ndarray):
        assert self.check_fitted()

        predictions = np.zeros_like(x)
        predictions[x >= self.__high] = 1
        predictions[x <= self.__low] = 1

        return predictions.astype(np.int).reshape(-1)
