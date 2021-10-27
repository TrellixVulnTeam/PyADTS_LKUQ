"""
@Time    : 2021/10/25 11:51
@File    : esd.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesRepository


class GeneralizedESD(Detector):
    def __init__(self):
        super(GeneralizedESD, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeriesRepository], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesRepository]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeriesRepository]):
        pass
