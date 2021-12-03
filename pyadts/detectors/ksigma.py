"""
@Time    : 2021/10/25 11:52
@File    : ksigma.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesDataset


class KSigmaDetector(Detector):
    def __init__(self):
        super(KSigmaDetector, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeriesDataset], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass
