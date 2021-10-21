"""
@Time    : 2021/10/18 11:11
@File    : window.py
@Software: PyCharm
@Desc    : 
"""

import numpy as np

from pyadts.utils import sliding_window_with_stride


def window_mean(x: np.ndarray, window_size: int, stride: int):
    x_window = sliding_window_with_stride(x, window_size, stride)

    return np.mean(x_window, axis=-1)


def window_var(x: np.ndarray, window_size: int, stride: int):
    x_window = sliding_window_with_stride(x, window_size, stride)

    return (np.std(x_window, axis=-1)) ** 2
