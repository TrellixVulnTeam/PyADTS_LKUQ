"""
@Time    : 2021/10/25 11:03
@File    : gradient.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesRepository


class GradientDetector(Detector):
    def __init__(self, max_gradient: float = np.inf):
        super(GradientDetector, self).__init__()

        self.max_gradient = max_gradient

    def fit(self, x: Union[np.ndarray, TimeSeriesRepository], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesRepository]):
        if isinstance(x, TimeSeriesRepository):
            x = x.data

    def score(self, x: Union[np.ndarray, TimeSeriesRepository]):
        if isinstance(x, TimeSeriesRepository):
            x = x.data
