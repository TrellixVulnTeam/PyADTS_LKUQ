"""
@Time    : 2021/10/25 11:56
@File    : regression.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np
import torch
from statsmodels.tsa.arima_model import ARIMA

from pyadts.generic import Detector, TimeSeriesDataset
from pyadts.utils.data import any_to_numpy


class RegressionResidualDetector(Detector):
    def __init__(self, regressor: str):
        super(RegressionResidualDetector, self).__init__()

        self.regressor = regressor

    def fit(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], y=None):
        pass

    def score(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor]):
        x = any_to_numpy(x)

        if self.regressor == 'arima':
            model = ARIMA()
        else:
            raise ValueError

        model_fit = model.fit()
        return model_fit.resid
