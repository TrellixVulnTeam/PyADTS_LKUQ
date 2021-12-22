"""
@Time    : 2021/10/25 10:59
@File    : transform.py
@Software: PyCharm
@Desc    : 
"""
import abc
from typing import Union

import numpy as np
import torch


class Transform(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        pass
