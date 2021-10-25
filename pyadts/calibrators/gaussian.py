"""
@Time    : 2021/10/25 15:19
@File    : gaussian.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from pyadts.generic import Function


class GaussianCalibrator(Function):
    def __init__(self):
        super(GaussianCalibrator, self).__init__()

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray):
        pass

    def fit_transform(self, x: np.ndarray):
        pass
