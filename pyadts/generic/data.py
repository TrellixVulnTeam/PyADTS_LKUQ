"""
@Time    : 2021/10/24 11:25
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import abc
from typing import Union, List, Tuple

import numpy as np
import torch

from pyadts.utils.visualization import plot_series


class TimeSeriesDataset(abc.ABC):
    def __init__(self, data: List[np.ndarray] = None, labels: np.ndarray = None):
        self.data = data
        self.labels = labels
        self.sep_indicators = None

    def windowed_view(self):
        pass

    def subsequence_view(self):
        pass

    def flatten_view(self):
        pass

    def numpy(self) -> Tuple[np.ndarray]:
        return self.data.transpose(), self.labels

    def tensor(self) -> Tuple[torch.Tensor]:
        return torch.from_numpy(self.data.transpose().astype(np.float32)), torch.from_numpy(self.labels.astype(np.long))

    def plot(self, series_id: int = 0, channel_id: int = 0, show: bool = True):
        assert series_id < self.num_series
        assert channel_id < self.num_channels

        fig = plot_series(self.get_series(series_id, channel_id))
        if show:
            fig.show()

        return fig

    def get_series(self, series_id: int, channel_id: Union[int, List[int]] = None):
        assert series_id < self.num_series
        if isinstance(channel_id, int):
            assert channel_id < self.num_channels
        if isinstance(channel_id, list):
            for c in channel_id:
                assert c < self.num_channels

        if channel_id is None:
            channel_id = list(range(self.num_channels))

        if series_id == 0:
            return self.data[channel_id, :self.sep_indicators[series_id]]
        else:
            return self.data[channel_id, self.sep_indicators[series_id - 1]: self.sep_indicators[series_id]]

    @property
    def num_channels(self):
        return self.data.shape[0]

    @property
    def num_series(self):
        return len(self.sep_indicators)

    @property
    def num_timestamps(self):
        return self.data.shape[-1]
