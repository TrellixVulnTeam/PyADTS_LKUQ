"""
@Time    : 2021/10/18 11:33
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
import abc
from typing import Union, IO

import numpy as np


class Function(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def transform(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def fit_transform(self, x: np.ndaray):
        pass


class Model(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray = None):
        pass

    @abc.abstractmethod
    def predict(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def score(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def save(self, f: Union[str, IO]):
        pass

    @abc.abstractmethod
    def load(self, f: Union[str, IO]):
        pass
