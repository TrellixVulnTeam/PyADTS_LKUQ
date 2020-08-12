from typing import Union

import numpy as np

from statsmodels.tsa.seasonal import STL
from interpacf import interpolated_acf, dominant_period


def __auto_period(value: np.ndarray):
    lag, acf = interpolated_acf(np.arange(value.shape[0]), value)
    period = dominant_period(lag, acf, plot=False)

    return int(period)


def get_stl_feature(value: np.ndarray, period: Union[int, str]='auto', seasonal: int=7, robust: bool=True) -> np.ndarray:
    if isinstance(period, str):
        assert period == 'auto'

    if period == 'auto':
        period = __auto_period(value)
        if np.isnan(period):
            raise ValueError('No period found!')

    stl = STL(value, seasonal=seasonal, period=period, robust=robust)
    res = stl.fit()
    feature = np.transpose(np.stack((res.trend, res.seasonal, res.resid)))

    return feature
