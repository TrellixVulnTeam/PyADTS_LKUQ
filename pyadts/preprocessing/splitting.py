"""
@Time    : 2021/10/22 11:38
@File    : splitting.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple, Union

import numpy as np
import torch

from pyadts.generic import TimeSeriesDataset


def __train_test_split_series(data: TimeSeriesDataset, train_ratio: float, method: str = 'point',
                              shuffle: bool = True) -> Tuple[
    TimeSeriesDataset, TimeSeriesDataset]:
    """

    Args:
        data ():
        train_ratio ():
        method ():
        shuffle ():

    Returns:

    """

    if method == 'series':
        counts = data.num_series
        train_size = int(counts * train_ratio)
        if shuffle:
            train_idx = np.random.choice(np.arange(counts), train_size, replace=False)
        else:
            train_idx = np.arange(train_size)
        test_idx = np.setdiff1d(np.arange(counts), train_idx)

        train_dfs = []
        for i in train_idx:
            train_dfs.append(data.dfs[i])
        train_dataset = TimeSeriesDataset.from_iterable(train_dfs)

        test_dfs = []
        for i in test_idx:
            test_dfs.append(data.dfs[i])
        test_dataset = TimeSeriesDataset.from_iterable(test_dfs)
    elif method == 'point':
        train_dfs = []
        test_dfs = []

        counts = data.num_points
        train_size = int(counts * train_ratio)
        if shuffle:
            train_idx = np.random.choice(np.arange(counts), train_size, replace=False)
        else:
            train_idx = np.arange(train_size)
        test_idx = np.setdiff1d(np.arange(counts), train_idx)

        sum_size = 0
        for df in data.dfs:
            train_df = df.iloc[train_idx[train_idx >= sum_size]]
            train_df = train_df.reset_index(drop=True)
            train_dfs.append(train_df)

            test_df = df.iloc[test_idx[test_idx >= sum_size]]
            test_df = test_df.reset_index(drop=True)
            test_dfs.append(test_df)

            sum_size += df.shape[0]
    elif method == 'point_balanced':
        train_dfs = []
        test_dfs = []

        for df in data.dfs:
            counts = df.shape[0]
            train_size = int(counts * train_ratio)
            if shuffle:
                train_idx = np.random.choice(np.arange(counts), train_size, replace=False)
            else:
                train_idx = np.arange(train_size)
            test_idx = np.setdiff1d(np.arange(counts), train_idx)
            train_df = df.iloc[train_idx]
            train_df = train_df.reset_index(drop=True)
            train_dfs.append(train_df)
            test_df = df.iloc[test_idx]
            test_df = test_df.reset_index(drop=True)
            test_dfs.append(test_df)

        train_dataset = TimeSeriesDataset.from_iterable(train_dfs)
        test_dataset = TimeSeriesDataset.from_iterable(test_dfs)
    else:
        raise ValueError

    return train_dataset, test_dataset


def __train_test_split_tensors(x: Union[np.ndarray, torch.Tensor], *vals: Tuple[Union[np.ndarray, torch.Tensor]],
                               train_ratio: float = 0.5, shuffle: bool = True):
    for val in vals:
        assert len(val) == len(x) and type(x) == type(val)

    indices = np.arange(len(x))

    if shuffle:
        np.random.shuffle(indices)

    train_size = int(len(indices) * train_ratio)
    train_indices = indices[: train_size]
    test_indices = np.setdiff1d(indices, train_indices)

    train_x = x[train_indices]
    train_vals = (val[train_indices] for val in vals)

    test_x = x[test_indices]
    test_vals = (val[test_indices] for val in vals)

    return (train_x, *train_vals), (test_x, *test_vals)


class TrainTestSplitter(object):
    def __init__(self, train_ratio: float, split_on: str = 'point', shuffle: bool = True):
        assert split_on in ['point', 'series']

        self.train_ratio = train_ratio
        self.split_on = split_on
        self.shuffle = shuffle

    def __call__(self, x: Union[np.ndarray, torch.Tensor, TimeSeriesDataset], *args, **kwargs):
        if isinstance(x, TimeSeriesDataset):
            pass
        elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            vals = tuple(filter(lambda val: type(val) == type(x), args))
        else:
            raise ValueError
