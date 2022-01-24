"""
@Time    : 2021/10/24 11:23
@File    : detectors.py
@Software: PyCharm
@Desc    :
"""
import abc
import os
from pathlib import Path
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
        raise NotImplementedError

    def predict(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], calibrator: Type[Calibrator], *args,
                **kwargs):
        scores = self.score(x)

        return calibrator(*args, **kwargs).calibrate(scores)

    @abc.abstractmethod
    def score(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor]):
        raise NotImplementedError

    def save(self, f: Union[str, Path, IO], creat_folder: bool = True):
        """
        Write model states into a file.

        Args:
            f:
            creat_folder:

        Returns:
            none
        """
        # print(self.__dict__)
        if isinstance(f, str):
            parent_folder, _ = os.path.split(f)
            if creat_folder and not os.path.exists(parent_folder):
                os.makedirs(parent_folder)
        elif isinstance(f, Path):
            parent_folder = f.parent
            if creat_folder and not parent_folder.exists():
                parent_folder.mkdir(exist_ok=False)

        save_objects(self.__dict__, f)

    def load(self, f: Union[str, Path, IO]):
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
