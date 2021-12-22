"""
@Time    : 2021/10/25 11:52
@File    : sr.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesDataset

EPS = 1e-8


def __average_filter(x: np.ndarray, n: int = 3):
    if n >= len(x):
        n = len(x)

    res = np.cumsum(x, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res


def __spectral_residual_transform(x: np.ndarray, window_size: int = 5):
    assert x.ndim == 1

    trans = np.fft.fft(x)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
    eps_index = np.where(mag <= EPS)[0]
    mag[eps_index] = EPS

    mag_log = np.log(mag)
    mag_log[eps_index] = 0

    spectral = np.exp(mag_log - __average_filter(mag_log, n=window_size))

    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
    return mag


class SpectralResidual(Detector):
    def __init__(self):
        super(SpectralResidual, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeriesDataset], y=None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass
