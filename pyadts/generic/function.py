"""
@Time    : 2021/10/25 10:59
@File    : function.py
@Software: PyCharm
@Desc    : 
"""
import abc

import numpy as np


class Function(abc.ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    @abc.abstractmethod
    def fit(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def transform(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def fit_transform(self, x: np.ndarray):
        pass
