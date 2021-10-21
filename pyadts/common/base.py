"""
@Time    : 2021/10/18 11:33
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
import abc
from typing import Union

import numpy as np


class Data(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __getitem(self, item):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class Function(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, x: Union[np.ndarray, Data]):
        pass

    @abc.abstractmethod
    def transform(self, x: Union[np.ndarray, Data]):
        pass

    @abc.abstractmethod
    def fit_transform(self, x: Union[np.ndarray, Data]):
        pass


class Model(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, x: Union[np.ndarray, Data], y: np.ndarray = None):
        pass

    @abc.abstractmethod
    def predict(self, x: Union[np.ndarray, Data]):
        pass

    @abc.abstractmethod
    def score(self, x: Union[np.ndarray, Data]):
        pass
