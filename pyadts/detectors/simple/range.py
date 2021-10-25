from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeries


class RangeDetector(Detector):
    def __init__(self, low: float, high: float):
        super(RangeDetector, self).__init__()

        self.low = low
        self.high = high

        if self.low >= self.high:
            raise ValueError('The parameter `high` must be larger than `low`!')

    def fit(self, x: Union[np.ndarray, TimeSeries], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeries]):
        if isinstance(x, TimeSeries):
            x = x.data

        predictions = np.zeros_like(x)
        predictions[x > self.high] = 1
        predictions[x < self.low] = 1

        return predictions.astype(np.int).reshape(-1)

    def score(self, x: Union[np.ndarray, TimeSeries]):
        if isinstance(x, TimeSeries):
            x = x.data

        return np.abs(x - (self.low + self.high) / 2).reshape(-1)
