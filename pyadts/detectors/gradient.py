"""
@Time    : 2021/10/25 11:03
@File    : gradient.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesDataset
from pyadts.utils.data import any_to_numpy


class GradientDetector(Detector):
    def __init__(self, max_gradient: float = np.inf):
        super(GradientDetector, self).__init__()

        self.max_gradient = max_gradient

    def fit(self, x: Union[np.ndarray, TimeSeriesDataset], y=None):
        pass

    # def predict(self, x: Union[np.ndarray, TimeSeriesDataset]):
    #     if isinstance(x, TimeSeriesDataset):
    #         x = x.to_numpy()

    def score(self, x: Union[np.ndarray, TimeSeriesDataset]):
        x = any_to_numpy(x)

        x_diff = np.diff(x, prepend=x[0])

        return np.abs(x_diff)
