"""
@Time    : 2021/12/4 1:30
@File    : cross_validation.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple, Iterable

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut

from pyadts.generic import TimeSeriesDataset


def leave_one_out(data: TimeSeriesDataset) -> Iterable[Tuple[TimeSeriesDataset, TimeSeriesDataset]]:
    """
    **Leave One Out** cross validation for `TimeSeriesDataset`.

    Args:
        data:

    Returns:

    """
    splitter = LeaveOneOut()
    counts = data.num_series

    for train_idx, test_idx in splitter.split(np.arange(counts)):
        train_dfs = []
        for i in train_idx:
            train_dfs.append(data.dfs[i])
        train_dataset = TimeSeriesDataset.from_iterable(train_dfs)

        test_dfs = []
        for i in test_idx:
            test_dfs.append(data.dfs[i])
        test_dataset = TimeSeriesDataset.from_iterable(test_dfs)

        yield train_dataset, test_dataset


def kfold_cross_validation(data: TimeSeriesDataset, method: str = 'point', kfolds: int = 10, shuffle: bool = False):
    if method == 'series':
        counts = data.num_series
        assert counts >= kfolds
        splitter = KFold(n_splits=kfolds)

        for train_idx, test_idx in splitter.split(np.arange(counts), shuffle=shuffle):
            train_dfs = []
            for i in train_idx:
                train_dfs.append(data.dfs[i])
            train_dataset = TimeSeriesDataset.from_iterable(train_dfs)

            test_dfs = []
            for i in test_idx:
                test_dfs.append(data.dfs[i])
            test_dataset = TimeSeriesDataset.from_iterable(test_dfs)

            yield train_dataset, test_dataset
    elif method == 'point':
        counts = data.num_points
        assert counts >= kfolds
        splitter = KFold(n_splits=kfolds)

        for train_idx, test_idx in splitter.split(np.arange(counts), shuffle=shuffle):
            train_dfs = []
            test_dfs = []

            sum_size = 0
            for df in data.dfs:
                train_df = df.iloc[train_idx[train_idx >= sum_size]]
                train_df = train_df.reset_index(drop=True)
                train_dfs.append(train_df)

                test_df = df.iloc[test_idx[test_idx >= sum_size]]
                test_df = test_df.reset_index(drop=True)
                test_dfs.append(test_df)

                sum_size += df.shape[0]

            train_dataset = TimeSeriesDataset.from_iterable(train_dfs)
            test_dataset = TimeSeriesDataset.from_iterable(test_dfs)

            yield train_dataset, test_dataset
    elif method == 'point_balanced':
        counts = data.num_points
        assert counts >= kfolds
        splitter = StratifiedKFold(n_splits=kfolds)
        series_ids = []
        for i, df in enumerate(data.dfs):
            series_ids += ([i] * df.shape[0])

        for train_idx, test_idx in splitter.split(np.arange(counts), series_ids, shuffle=shuffle):
            train_dfs = []
            test_dfs = []

            sum_size = 0
            for df in data.dfs:
                train_df = df.iloc[train_idx[train_idx >= sum_size]]
                train_df = train_df.reset_index(drop=True)
                train_dfs.append(train_df)

                test_df = df.iloc[test_idx[test_idx >= sum_size]]
                test_df = test_df.reset_index(drop=True)
                test_dfs.append(test_df)

                sum_size += df.shape[0]

            train_dataset = TimeSeriesDataset.from_iterable(train_dfs)
            test_dataset = TimeSeriesDataset.from_iterable(test_dfs)

            yield train_dataset, test_dataset
    else:
        raise ValueError
