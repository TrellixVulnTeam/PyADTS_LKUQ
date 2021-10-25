"""
@Time    : 2021/10/25 15:22
@File    : conformal.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from pyadts.generic import Function


class ConformalCalibrator(Function):
    def __init__(self):
        super(ConformalCalibrator, self).__init__()

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray):
        pass

    def fit_transform(self, x: np.ndarray):
        pass
