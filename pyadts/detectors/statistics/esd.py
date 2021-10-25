"""
@Time    : 2021/10/25 11:51
@File    : esd.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeries


class GeneralizedESD(Detector):
    def __init__(self):
        super(GeneralizedESD, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeries], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeries]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeries]):
        pass
