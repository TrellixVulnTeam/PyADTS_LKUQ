"""
@Time    : 2021/10/26 0:15
@File    : iforest.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np
from sklearn.ensemble import IsolationForest

from pyadts.generic import Detector, TimeSeriesDataset


class IForest(Detector):
    def __init__(self, **kwargs):
        super(IForest, self).__init__()

        self.model = IsolationForest(**kwargs)

    def fit(self, x: Union[np.ndarray, TimeSeriesDataset], y=None):
        if isinstance(x, TimeSeriesDataset):
            x = x.to_numpy()

        self.model.fit(x)

    def score(self, x: Union[np.ndarray, TimeSeriesDataset]):
        if isinstance(x, TimeSeriesDataset):
            x = x.to_numpy()

        return self.model.decision_function(x)
