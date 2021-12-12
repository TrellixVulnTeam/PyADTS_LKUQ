"""
@Time    : 2021/12/3 17:33
@File    : test_data.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from pyadts.generic import TimeSeriesDataset


def test_constructor():
    length = 1000
    num_series = 10
    num_channel = 5
    timestamps = np.arange(1000)
    data = []
    timestamps = []

    for i in range(num_series):
        data_current = []
        for c in range(num_channel):
            idx = np.arange(length - i)
            data_current.append(np.sin(idx) * 1 / (i + c) + np.cos(idx) * 1 / (i + c))
        data_current = np.stack(data_current, axis=-1)
        data.append(data_current)
        timestamps.append(np.arange(length - i))

    ts_data = TimeSeriesDataset(data, timestamps)
    print(ts_data)

    print('Numpy data shape:', ts_data.data(return_format='numpy').shape)
    print('Tensor data shape:', ts_data.data(return_format='tensor').shape)
    print('Numpy targets shape:', ts_data.targets(return_format='numpy').shape)
    print('Tensor targets shape:', ts_data.targets(return_format='tensor').shape)
