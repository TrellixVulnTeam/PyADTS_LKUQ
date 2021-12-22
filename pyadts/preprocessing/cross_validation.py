"""
@Time    : 2021/12/4 1:30
@File    : cross_validation.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple, Iterable, Union

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut

from pyadts.generic import TimeSeriesDataset


def leave_one_out_series(data: TimeSeriesDataset) -> Iterable[Tuple[TimeSeriesDataset, TimeSeriesDataset]]:
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


def leave_one_out_tensors(x: Union[np.ndarray, torch.Tensor], *vals: Tuple[Union[np.ndarray, torch.Tensor]]):
    for val in vals:
        assert len(val) == len(x) and type(x) == type(val)

    indices = np.arange(len(x))
    splitter = LeaveOneOut()

    for train_indices, test_indices in splitter.split(indices):
        train_x = x[train_indices]
        train_vals = (val[train_indices] for val in vals)

        test_x = x[test_indices]
        test_vals = (val[test_indices] for val in vals)

        yield (train_x, *train_vals), (test_x, *test_vals)


def kfold_series(data: TimeSeriesDataset, method: str = 'point', kfolds: int = 10, shuffle: bool = False):
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


def kfold_tensors(x: Union[np.ndarray, torch.Tensor], *vals: Tuple[Union[np.ndarray, torch.Tensor]], kfolds: int = 10,
                  shuffle: bool = False):
    for val in vals:
        assert len(val) == len(x) and type(x) == type(val)

    indices = np.arange(len(x))
    assert len(indices) >= kfolds
    splitter = KFold(n_splits=kfolds)

    for train_indices, test_indices in splitter.split(indices, shuffle=shuffle):
        train_x = x[train_indices]
        train_vals = (val[train_indices] for val in vals)

        test_x = x[test_indices]
        test_vals = (val[test_indices] for val in vals)

        yield (train_x, *train_vals), (test_x, *test_vals)


class LeaveOneOutValidator(object):
    def __init__(self):
        pass

    def __call__(self, x: Union[np.ndarray, torch.Tensor, TimeSeriesDataset], *args, **kwargs):
        if isinstance(x, TimeSeriesDataset):
            for train_vals, test_vals in leave_one_out_series(x):
                yield train_vals, test_vals
        elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            vals = tuple(filter(lambda val: type(val) == type(x), args))
            for train_vals, test_vals in leave_one_out_tensors(x, *vals):
                yield train_vals, test_vals
        else:
            raise ValueError


class KFoldValidator(object):
    def __init__(self):
        pass

    def __call__(self, x: Union[np.ndarray, torch.Tensor, TimeSeriesDataset], *args, **kwargs):
        shuffle = kwargs.get('shuffle', False)
        kfolds = kwargs.get('kfolds', 10)

        if isinstance(x, TimeSeriesDataset):
            for train_vals, test_vals in kfold_series(x, kfolds=kfolds, shuffle=shuffle):
                yield train_vals, test_vals
        elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            vals = tuple(filter(lambda val: type(val) == type(x), args))
            for train_vals, test_vals in kfold_tensors(x, *vals, kfolds=kfolds, shuffle=shuffle):
                yield train_vals, test_vals
        else:
            raise ValueError
