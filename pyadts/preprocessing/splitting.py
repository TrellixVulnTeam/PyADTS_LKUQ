"""
@Time    : 2021/10/22 11:38
@File    : splitting.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple, Iterable

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

from pyadts.generic import TimeSeriesDataset


def simple_split(data: TimeSeriesDataset, split_on: str, train_ratio: float) -> Tuple[
    TimeSeriesDataset, TimeSeriesDataset]:
    """
    Apply a simple train-test splitting on the specified `split_on` dimension.

    Args:
        data ():
        split_on ():
        train_ratio ():

    Returns:

    """
    if split_on == 'series':
        counts = data.num_series
        train_size = int(counts * train_ratio)
        train_idx = np.random.choice(np.arange(counts), train_size, replace=False)
        test_idx = np.setdiff1d(np.arange(counts), train_idx)
        train_dataset = TimeSeriesDataset([], [] if data.labels is not None else None)
        test_dataset = TimeSeriesDataset([], [] if data.labels is not None else None)
    elif split_on == 'sequence':
        train_dataset = TimeSeriesDataset([], [] if data.labels is not None else None)
        test_dataset = TimeSeriesDataset([], [] if data.labels is not None else None)
    elif split_on == 'instance':
        train_dataset = TimeSeriesDataset([], [] if data.labels is not None else None)
        test_dataset = TimeSeriesDataset([], [] if data.labels is not None else None)
    else:
        raise ValueError

    return train_dataset, test_dataset


def cross_validation(data: TimeSeriesDataset, method: str, split_on: str, kfolds: int = 10) -> Iterable[
    Tuple[TimeSeriesDataset, TimeSeriesDataset]]:
    """

    Args:
        data ():
        method (str): . choices: ['kfold', 'leave_one_out']
        split_on (str):
        kfolds (int):

    Returns:

    """
    if method == 'kfold':
        splitter = KFold(n_splits=kfolds)
    elif method == 'leave_one_out':
        splitter = LeaveOneOut()
    else:
        raise ValueError

    if split_on == 'series':
        pass
    elif split_on == 'sequence':
        pass
    elif split_on == 'instance':
        pass
    else:
        raise ValueError

    for train_idx, test_idx in splitter.split(data.data):
        pass


class SimpleSplit(object):
    def __init__(self, split_on: str, train_ratio: float):
        """

        Args:
            split_on ():
            train_ratio ():
        """
        pass

    def __call__(self, data: TimeSeriesDataset) -> Tuple[TimeSeriesDataset, TimeSeriesDataset]:
        """

        Args:
            data ():

        Returns:

        """
        pass


class CrossValidation(object):
    def __init__(self, method: str, split_on: str, kfolds: int = 10):
        """

        Args:
            method ():
            split_on ():
            kfolds ():
        """
        pass

    def __call__(self, data: TimeSeriesDataset) -> Iterable[Tuple[TimeSeriesDataset, TimeSeriesDataset]]:
        """

        Args:
            data ():

        Returns:

        """
        pass
