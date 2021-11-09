"""
@Time    : 2021/10/25 11:49
@File    : interquartile.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesDataset


class InterQuartileRangeDetector(Detector):
    def __init__(self):
        super(InterQuartileRangeDetector, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeriesDataset], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass
