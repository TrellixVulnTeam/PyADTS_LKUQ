"""
@Time    : 2021/10/24 11:23
@File    : detectors.py
@Software: PyCharm
@Desc    :
"""
import abc
from typing import Union, IO, Type

import numpy as np
import torch

from pyadts.utils.io import save_objects, load_objects
from .calibrator import Calibrator
from .data import TimeSeriesDataset


class Detector(abc.ABC):
    def __int__(self):
        pass

    @abc.abstractmethod
    def fit(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor] = None):
        pass

    def predict(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], calibrator: Type[Calibrator], *args,
                **kwargs):
        scores = self.score(x)

        return calibrator(*args, **kwargs).calibrate(scores)

    @abc.abstractmethod
    def score(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor]):
        pass

    def save(self, f: Union[str, IO]):
        """
        Write model states into a file.

        Args:
            f:

        Returns:
            none
        """
        # print(self.__dict__)
        save_objects(self.__dict__, f)

    def load(self, f: Union[str, IO]):
        """
        Load states from a file.

        Args:
            f:

        Returns:
            none
        """
        objs = load_objects(f)
        # print(objs)
        self.__dict__.update(objs)
