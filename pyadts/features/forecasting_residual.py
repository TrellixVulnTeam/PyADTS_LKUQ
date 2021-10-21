"""
@Time    : 2021/10/18 11:12
@File    : forecasting_residual.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from statsmodels.tsa.api import SARIMAX, ExponentialSmoothing, SimpleExpSmoothing, Holt


def sarima_residual(x: np.ndarray):
    predict = SARIMAX(x, trend='n').fit(disp=0).get_prediction()

    return x - predict.predicted_mean


def addes_residual(x: np.ndarray):
    predict = ExponentialSmoothing(x, trend='add').fit(smoothing_level=1)

    return x - predict.fittedvalues


def simplees_residual(x: np.ndarray):
    predict = SimpleExpSmoothing(x).fit(smoothing_level=1)

    return x - predict.fittedvalues


def holt_residual(x: np.ndarray):
    predict = Holt(x).fit(smoothing_level=1)

    return x - predict.fittedvalues
