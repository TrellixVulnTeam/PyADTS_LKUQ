"""
@Time    : 2021/10/27 18:49
@File    : matrix_profile.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np
import stumpy
import torch
from sklearn.preprocessing import MinMaxScaler

from pyadts.generic import Detector, TimeSeriesDataset
from pyadts.utils.data import any_to_numpy


class MatrixProfile(Detector):
    def __init__(self, window_size: int):
        super(MatrixProfile, self).__init__()

        self.window_size = window_size

    def fit(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], y=None) -> None:
        pass

    def score(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor]) -> np.ndarray:
        x = any_to_numpy(x).astype(np.float64)  # (num_points, num_features)

        mp, mp_idx = stumpy.mstump(x.transpose(), m=self.window_size)

        mp = np.concatenate((mp, np.repeat(mp[:, -1:], repeats=self.window_size - 1, axis=-1)), axis=-1)

        scaler = MinMaxScaler()
        mp = scaler.fit_transform(mp)
        mp = np.sum(mp.transpose(), axis=-1)

        return mp
