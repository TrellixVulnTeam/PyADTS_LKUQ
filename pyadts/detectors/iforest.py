"""
@Time    : 2021/10/26 0:15
@File    : iforest.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from pyadts.generic import Detector, TimeSeriesDataset
from pyadts.utils.data import any_to_numpy


class IForest(Detector):
    def __init__(self, **kwargs):
        super(IForest, self).__init__()

        self.model = IsolationForest(**kwargs)

    def fit(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], y=None):
        x = any_to_numpy(x)

        self.model.fit(x)

    def score(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor]):
        x = any_to_numpy(x)

        return self.model.decision_function(x)
