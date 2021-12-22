from typing import Union, Type

import numpy as np
import torch

from pyadts.generic import Detector, TimeSeriesDataset, Calibrator


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
        if isinstance(x, TimeSeriesDataset):
            x = x.to_numpy()
        elif isinstance(x, torch.Tensor):
            x = x.numpy()

        predictions = np.zeros_like(x)
        predictions[x > self.high] = 1
        predictions[x < self.low] = 1

        return predictions.astype(np.int).reshape(-1)

    def score(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor]):
        if isinstance(x, TimeSeriesDataset):
            x = x.to_numpy()
        elif isinstance(x, torch.Tensor):
            x = x.numpy()

        return np.abs(x - (self.low + self.high) / 2).mean(axis=-1)
