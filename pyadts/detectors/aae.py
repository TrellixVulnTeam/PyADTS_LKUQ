"""
@Time    : 2021/10/26 0:14
@File    : aae.py
@Software: PyCharm
@Desc    :
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesDataset


class AAE(Detector):
    def __init__(self):
        super(AAE, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeriesDataset], y=None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass
