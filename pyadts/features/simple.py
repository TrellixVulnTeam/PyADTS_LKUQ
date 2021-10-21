"""
@Time    : 2021/10/18 11:11
@File    : simple.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np


def logarithm(value: np.ndarray) -> np.ndarray:
    return np.log(value)


def differential(value: np.ndarray) -> np.ndarray:
    return np.diff(value, prepend=0)


def differential_second_order(value: np.ndarray) -> np.ndarray:
    return np.diff(np.diff(value, prepend=0), prepend=0)
