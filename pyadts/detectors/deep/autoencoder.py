"""
@Time    : 2021/10/25 15:13
@File    : autoencoder.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeries


class Autoencoder(Detector):
    def __init__(self):
        super(Autoencoder, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeries], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeries]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeries]):
        pass
