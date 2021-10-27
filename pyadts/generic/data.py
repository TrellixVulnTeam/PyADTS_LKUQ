"""
@Time    : 2021/10/24 11:25
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import abc

import numpy as np


class TimeSeriesRepository(abc.ABC):
    def __init__(self, data: np.ndarray = None, labels: np.ndarray = None, sep_indicators: np.ndarray = None):
        self.data = data
        self.labels = labels
        self.sep_indicators = sep_indicators

    def window_view(self):
        pass

    def flatten_view(self):
        pass

    def numpy(self):
        pass

    def tensor(self):
        pass

    def plot(self):
        pass

    @property
    def num_channels(self):
        return self.data.shape[0]

    @property
    def num_series(self):
        return len(self.sep_indicators)

    @property
    def num_timestamps(self):
        return self.data.shape[-1]

    # @property
    # def timestamps(self):
    #     return self.timestamps

    # @property
    # def anomalies(self):
    #     return self.anomalies
    #
    # @property
    # def missing_values(self):
    #     return self.missing_values
