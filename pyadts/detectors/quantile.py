"""
@Time    : 2021/10/25 11:38
@File    : quantile.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesDataset


class QuantileDetector(Detector):
    def __init__(self, low=0.01, high=0.99):
        super(QuantileDetector, self).__init__()

        self.low = low
        self.high = high

    def fit(self, x: Union[np.ndarray, TimeSeriesDataset], y=None):
        if isinstance(x, TimeSeriesDataset):
            x = x.to_numpy

    def predict(self, x: Union[np.ndarray, TimeSeriesDataset]):
        if isinstance(x, TimeSeriesDataset):
            x = x.to_numpy

    def score(self, x: Union[np.ndarray, TimeSeriesDataset]):
        if isinstance(x, TimeSeriesDataset):
            x = x.to_numpy
