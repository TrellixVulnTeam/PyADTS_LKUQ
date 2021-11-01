"""
@Time    : 2021/10/25 11:52
@File    : spot.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesRepository


class SPOT(Detector):
    def __init__(self, q: float = 1e-4):
        super(SPOT, self).__init__()

        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def fit(self, x: Union[np.ndarray, TimeSeriesRepository], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesRepository]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeriesRepository]):
        pass


class DSPOT(Detector):
    def __init__(self):
        super(DSPOT, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeriesRepository], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesRepository]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeriesRepository]):
        pass
