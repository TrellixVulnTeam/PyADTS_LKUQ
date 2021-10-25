"""
@Time    : 2021/10/25 15:12
@File    : usad.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeries


class USAD(Detector):
    def __init__(self):
        super(USAD, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeries], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeries]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeries]):
        pass
