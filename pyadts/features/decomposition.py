"""
@Time    : 2021/10/18 11:12
@File    : decomposition.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np
from statsmodels.tsa.seasonal import STL

from pyadts.utils import auto_period


def stl_decomposition(x: np.ndarray, period: Union[int, str] = 'auto', seasonal: int = 7, robust: bool = True):
    if isinstance(period, str):
        assert period == 'auto'

    if period == 'auto':
        period = auto_period(x)

        if np.isnan(period):
            raise ValueError('No period found!')

    stl = STL(x, seasonal=seasonal, period=period, robust=robust)
    res = stl.fit()
    feature = np.transpose(np.stack((res.trend, res.seasonal, res.resid)))

    return feature
