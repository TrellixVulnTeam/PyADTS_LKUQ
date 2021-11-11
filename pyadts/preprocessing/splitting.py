"""
@Time    : 2021/10/22 11:38
@File    : splitting.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple, Iterable

from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

from pyadts.generic import TimeSeriesDataset


def simple_split(data: TimeSeriesDataset, split_granularity: str, train_ratio: float) -> Tuple[TimeSeriesDataset]:
    if split_granularity == 'series':
        pass
    elif split_granularity == 'window':
        pass
    elif split_granularity == 'instance':
        pass
    else:
        raise ValueError

    x_train, x_test, y_train, y_test = train_test_split()


def cross_validation(data: TimeSeriesDataset, method: str, split_granularity: str, kfolds: int = 10) -> Iterable[
    Tuple[TimeSeriesDataset]]:
    if method == 'kfolds':
        splitter = KFold(n_splits=kfolds)
    elif method == 'leave_one_out':
        splitter = LeaveOneOut()
    else:
        raise ValueError

    if split_granularity == 'series':
        pass
    elif split_granularity == 'window':
        pass
    elif split_granularity == 'instance':
        pass
    else:
        raise ValueError

    for train_idx, test_idx in splitter.split(data.data):
        pass


class SimpleSplit(object):
    def __init__(self, split_granularity: str, train_ratio: float):
        pass

    def __call__(self, data: TimeSeriesDataset) -> Tuple[TimeSeriesDataset]:
        pass


class CrossValidation(object):
    def __init__(self, method: str, split_granularity: str, kfolds: int = 10):
        pass

    def __call__(self, data: TimeSeriesDataset) -> Iterable[Tuple[TimeSeriesDataset]]:
        pass
