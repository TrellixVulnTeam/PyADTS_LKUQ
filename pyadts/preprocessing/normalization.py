"""
@Time    : 2021/10/18 11:09
@File    : normalization.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from pyadts.generic import Function


def robust_scale(x: np.ndarray):
    pass


def min_max_scale(x: np.ndarray):
    pass


def standard_scale(x: np.ndarray):
    pass


def quantile_scale(x: np.ndarray):
    pass


class RobustScaler(Function):
    def __init__(self):
        raise NotImplementedError


class MinMaxScaler(Function):
    def __init__(self):
        raise NotImplementedError


class StandardScaler(Function):
    def __init__(self):
        raise NotImplementedError


class QuantileScaler(Function):
    def __init__(self):
        raise NotImplementedError
