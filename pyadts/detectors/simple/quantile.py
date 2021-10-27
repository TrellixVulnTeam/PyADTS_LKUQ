"""
@Time    : 2021/10/25 11:38
@File    : quantile.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesRepository


class QuantileDetector(Detector):
    def __init__(self, low=0.01, high=0.99):
        super(QuantileDetector, self).__init__()

        self.low = low
        self.high = high

    def fit(self, x: Union[np.ndarray, TimeSeriesRepository], y: np.ndarray = None):
        if isinstance(x, TimeSeriesRepository):
            x = x.data

    def predict(self, x: Union[np.ndarray, TimeSeriesRepository]):
        if isinstance(x, TimeSeriesRepository):
            x = x.data

    def score(self, x: Union[np.ndarray, TimeSeriesRepository]):
        if isinstance(x, TimeSeriesRepository):
            x = x.data
