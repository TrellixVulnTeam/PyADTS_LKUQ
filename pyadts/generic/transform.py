"""
@Time    : 2021/10/25 10:59
@File    : transform.py
@Software: PyCharm
@Desc    : 
"""
import abc

import numpy as np


class Transform(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, x: np.ndarray):
        pass
