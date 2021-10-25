"""
@Time    : 2021/10/25 11:03
@File    : gradient.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeries


class GradientDetector(Detector):
    def __init__(self, max_gradient: float = np.inf):
        super(GradientDetector, self).__init__()

        self.max_gradient = max_gradient

    def fit(self, x: Union[np.ndarray, TimeSeries], y: np.ndarray = None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeries]):
        if isinstance(x, TimeSeries):
            x = x.data

    def score(self, x: Union[np.ndarray, TimeSeries]):
        if isinstance(x, TimeSeries):
            x = x.data
