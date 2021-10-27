"""
@Time    : 2021/10/26 0:15
@File    : iforest.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesRepository


class IsolationForest(Detector):
    def __init__(self):
        super(IsolationForest, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeriesRepository], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesRepository]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeriesRepository]):
        pass
