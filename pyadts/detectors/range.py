from typing import Union, Type

import numpy as np
import torch

from pyadts.generic import Detector, TimeSeriesDataset, Calibrator
from pyadts.utils.data import any_to_numpy


class RangeDetector(Detector):
    def __init__(self, low: float, high: float):
        super(RangeDetector, self).__init__()

        self.low = low
        self.high = high

        if self.low >= self.high:
            raise ValueError('The parameter `high` must be larger than `low`!')

    def fit(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor] = None):
        pass

    def predict(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], calibrator: Type[Calibrator], *args,
                **kwargs):
        x = any_to_numpy(x)

        predictions = np.zeros_like(x)
        predictions[x > self.high] = 1
        predictions[x < self.low] = 1

        return predictions.astype(np.int).reshape(-1)

    def score(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor]):
        x = any_to_numpy(x)

        return np.abs(x - (self.low + self.high) / 2).mean(axis=-1)
