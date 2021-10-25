"""
@Time    : 2021/10/22 17:33
@File    : sliding_window.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from pyadts.generic import Transform
from pyadts.utils import sliding_window_with_stride


class SlidingWindow(Transform):
    def __init__(self, window_size: int, stride: int):
        super(SlidingWindow, self).__init__()
        self.window_size = window_size
        self.stride = stride

    def __call__(self, x: np.ndarray):
        return sliding_window_with_stride(x, window_size=self.window_size, stride=self.stride)
