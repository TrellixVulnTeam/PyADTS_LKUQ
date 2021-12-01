"""
@Time    : 2021/10/18 11:09
@File    : normalization.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from pyadts.generic import Function, TimeSeriesDataset

EPS = 1e-6


def robust_scale(data: TimeSeriesDataset):
    raise NotImplementedError


def min_max_scale(data: TimeSeriesDataset):
    for series in data.data:
        min_val = np.min(series, axis=-1, keepdims=True)
        max_val = np.max(series, axis=-1, keepdims=True)
        series = (series - min_val) / (max_val - min_val + EPS)


def standard_scale(data: TimeSeriesDataset):
    for series in data.data:
        mean_val = np.mean(series, axis=-1, keepdims=True)
        std_val = np.std(series, axis=-1, keepdims=True)
        series = (series - mean_val) / (std_val + EPS)


def quantile_scale(data: TimeSeriesDataset):
    raise NotImplementedError


class RobustScaler(object):
    def __init__(self):
        pass

    def __call__(self, data: TimeSeriesDataset):
        return robust_scale(data)


class MinMaxScaler(object):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, data: TimeSeriesDataset):
        return min_max_scale(data)


class StandardScaler(object):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, data: TimeSeriesDataset):
        return standard_scale(data)


class QuantileScaler(Function):
    def __init__(self):
        raise NotImplementedError
