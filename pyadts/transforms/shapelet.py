"""
@Time    : 2021/10/18 11:43
@File    : shapelet.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union, List, Iterable

import numpy as np
from numba import njit

from pyadts.common import Data, Function
from pyadts.utils import sliding_window_with_stride


@njit()
def __extract_shapelets(x: np.ndarray, window_sizes: List[int], strides: List[int]):
    assert len(window_sizes) == len(strides)

    for w, s in zip(window_sizes, strides):
        x_strided = sliding_window_with_stride(x, w, s, copy=True)


def shapelet_transform(x: np.ndarray, num_shapelets: int, window_sizes: Iterable[int], strides: Iterable[int],
                       criterion: str = 'mutual_info', remove_similar: bool = True, sort: bool = True,
                       n_jobs: int = None):
    assert criterion in ['mutual_info', 'anova']

    window_sizes = list(window_sizes)
    strides = list(strides)


class ShapeletTransform(Function):
    def __init__(self):
        super(ShapeletTransform, self).__init__()

    def fit(self, x: Union[np.ndarray, Data]):
        pass

    def transform(self, x: Union[np.ndarray, Data]):
        pass

    def fit_transform(self, x: Union[np.ndarray, Data]):
        pass
