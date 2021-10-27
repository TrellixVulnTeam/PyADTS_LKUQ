"""
@Time    : 2021/10/22 11:38
@File    : splitting.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple, Iterable

import numpy as np

from pyadts.generic import TimeSeriesRepository


def train_test_split(data: TimeSeriesRepository, method: str, split_granularity: str) -> Tuple[np.ndarray]:
    pass


def cross_validation(data: TimeSeriesRepository, method: str, split_granularity: str) -> Iterable[Tuple[np.ndarray]]:
    if method == 'kfolds':
        pass
    elif method == 'leave_one_out':
        pass
    else:
        raise ValueError
