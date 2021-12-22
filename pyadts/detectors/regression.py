"""
@Time    : 2021/10/25 11:56
@File    : regression.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesDataset


class RegressionResidualDetector(Detector):
    def __init__(self):
        super(RegressionResidualDetector, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeriesDataset], y=None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass
