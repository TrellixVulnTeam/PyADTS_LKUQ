"""
@Time    : 2021/10/25 15:46
@File    : ndt.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from pyadts.generic import Function


class NDTCalibrator(Function):
    """
    Implementation of Nonparametric dynamic thresholding
    """

    def __init__(self):
        super(NDTCalibrator, self).__init__()

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray):
        pass

    def fit_transform(self, x: np.ndarray):
        pass
