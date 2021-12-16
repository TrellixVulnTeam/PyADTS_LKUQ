"""
@Time    : 2021/10/27 18:49
@File    : matrix_profile.py
@Software: PyCharm
@Desc    : 
"""

import numpy as np
import stumpy
from sklearn.preprocessing import MinMaxScaler

from pyadts.generic import Detector, TimeSeriesDataset


class MatrixProfile(Detector):
    def __init__(self, window_size: int):
        super(MatrixProfile, self).__init__()

        self.window_size = window_size

    def fit(self, x: TimeSeriesDataset):
        x = x.data()

        mp, mp_idx = stumpy.mstump(x.transpose(), m=self.window_size)

        blank = x.shape[0] - mp.shape[0]
        for i in range(blank):
            mp = np.append(mp, [mp[-1]], axis=0)

        scaler = MinMaxScaler()
        mp = scaler.fit_transform(mp)

        mp = np.sum(mp, axis=-1)

    def predict(self, x: TimeSeriesDataset):
        pass

    def score(self, x: TimeSeriesDataset):
        pass
