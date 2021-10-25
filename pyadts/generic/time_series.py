"""
@Time    : 2021/10/24 11:25
@File    : time_series.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from pyadts.utils.visualization import plot_series


class TimeSeries(object):
    def __init__(self, data: np.ndarray, timestamps: np.ndarray = None):
        if data.ndim == 1:
            self.data = data.reshape(1, -1)
        else:
            self.data = data

        self.timestamps = timestamps
        self.anomalies = None
        self.missing_values = None

    def plot(self, show: bool = True):
        fig = plot_series(self.data)

        if show:
            fig.show()

        return fig

    # @property
    # def data(self):
    #     return self.data

    @property
    def num_channels(self):
        return self.data.shape[-2]

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
