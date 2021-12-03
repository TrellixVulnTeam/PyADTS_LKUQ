"""
@Time    : 2021/10/24 11:25
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import abc
from typing import List, Tuple

import numpy as np
import seaborn as sns
import torch
from prettytable import PrettyTable

from pyadts.utils.data import sliding_window_with_stride
from pyadts.utils.visualization import plot_series


class TimeSeriesDataset(abc.ABC):
    def __init__(self, data: List[np.ndarray] = None, labels: List[np.ndarray] = None):
        """
        The abstract class of a time-series dataset. The dataset is  assumed to contain multiple
            multi-channel time-series, with the shape of `(*, C, T)`, where `C` is the number of channels,
            `T` is the number of time points.

        Args:
            data (List[np.ndarray], optional):
            labels (List[np.ndarray], optional):
        """
        if data is not None:
            for i in range(len(data)):
                assert data[i].shape[:-1] == data[0].shape[:-1]
                if labels is not None:
                    assert labels[i].shape == data[i][..., 0, :].shape

        self.data = data
        self.labels = labels

    def windowed_view(self, window_size: int, stride: int):
        """
        Applying a sliding window along the last dimension (`T`) with a window size `W` and a stride `S`.
            As a result, the last dimension of a time-series becomes `W`. And the number of windows is placed
            in the antepenultimate dimension. Namely, the shape of each time-series goes from `(*, C, T)`
            to `(*, N, C, W)` where `N` is the number of windows.

        Args:
            window_size ():
            stride ():

        Returns:
            (TimeSeriesDataset):
        """
        assert self.data is not None

        return TimeSeriesDataset(
            [np.swapaxes(sliding_window_with_stride(series, window_size, stride, copy=True), -2, -3)
             for series in self.data],
            [sliding_window_with_stride(label, window_size, stride, copy=True)
             for label in self.labels]
            if self.labels is not None else None)

    def merged_view(self):
        """

        Returns:

        """
        assert self.data is not None

        return TimeSeriesDataset(
            [np.concatenate(self.data, axis=-1)],
            [np.concatenate(self.labels, axis=-1)] if self.labels is not None else None
        )

    def numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """

        Returns:

        """
        return np.concatenate(self.data, axis=-1), \
               np.concatenate(self.labels, axis=-1) if self.labels is not None else None

    def tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Returns:

        """
        return torch.from_numpy(np.concatenate(self.data, axis=-1).astype(np.float32)), \
               torch.from_numpy(np.concatenate(self.labels, axis=-1).astype(np.long)) \
                   if self.labels is not None else None

    def plot(self, series_id: int = 0, channel_id: int = 0, show: bool = True):
        assert series_id < self.num_series
        assert channel_id < self.num_channels

        fig = plot_series(self.data[series_id][..., channel_id, :])
        if show:
            fig.show()

        return fig

    @property
    def num_channels(self):
        return self.data[0].shape[-2]

    @property
    def num_series(self):
        return len(self.data)

    @property
    def num_timestamps(self):
        return self.data[0].shape[-1]

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        table = PrettyTable()
        table.align = 'c'
        table.field_names = ['ID', '# Channels', '# Points', 'Anomaly Ratio', 'Missing Ratio']

        return table.get_string()
