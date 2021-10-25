"""
@Time    : 2021/10/25 15:12
@File    : omnianomaly.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeries


class OmniAnomaly(Detector):
    def __init__(self):
        pass

    def fit(self, x: Union[np.ndarray, TimeSeries], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeries]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeries]):
        pass
