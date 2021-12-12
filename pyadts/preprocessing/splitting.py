"""
@Time    : 2021/10/22 11:38
@File    : splitting.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple

import numpy as np

from pyadts.generic import TimeSeriesDataset


def train_test_split(data: TimeSeriesDataset, train_ratio: float, method: str = 'point', shuffle: bool = True) -> Tuple[
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

# class TrainTestSplitter(object):
#     def __init__(self, train_ratio: float, split_on: str = 'point', shuffle: bool = True):
#         assert split_on in ['point', 'series']
#
#         self.train_ratio = train_ratio
#         self.split_on = split_on
#         self.shuffle = shuffle
#
#     def __call__(self, data: TimeSeriesDataset) -> Tuple[TimeSeriesDataset, TimeSeriesDataset]:
#         return train_test_split(data, self.train_ratio, self.split_on, self.shuffle)
#
