"""
@Time    : 2021/10/24 11:23
@File    : detectors.py
@Software: PyCharm
@Desc    : 
"""
import abc
from typing import Union, IO

from pyadts.utils.io import save_objects, load_objects
from .data import TimeSeriesDataset


class Detector(abc.ABC):
    def __int__(self):
        pass

    @abc.abstractmethod
    def fit(self, x: TimeSeriesDataset):
        pass

    @abc.abstractmethod
    def predict(self, x: TimeSeriesDataset):
        pass

    @abc.abstractmethod
    def score(self, x: TimeSeriesDataset):
        pass

    def save(self, f: Union[str, IO]):
        """
        Write model states into a file.

        Args:
            f (str or IO):

        Returns:
            none
        """
        # print(self.__dict__)
        save_objects(self.__dict__, f)

    def load(self, f: Union[str, IO]):
        """
        Load states from a file.

        Args:
            f (str or IO):

        Returns:
            none
        """
        objs = load_objects(f)
        # print(objs)
        self.__dict__.update(objs)
