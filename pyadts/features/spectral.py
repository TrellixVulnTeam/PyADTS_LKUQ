"""
@Time    : 2021/10/18 11:11
@File    : spectral.py
@Software: PyCharm
@Desc    : 
"""
from typing import List

import numpy as np
import pywt


def __wavelet_decomposition(data, w, level=5):
    w = pywt.Wavelet(w)
    mode = pywt.Modes.smooth
    a = data
    ca = []
    cd = []
    for i in range(level):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i in range(level):
        rec_a.append(pywt.upcoef('a', ca[i], w, level=i + 1, take=data.shape[0]))
        rec_d.append(pywt.upcoef('d', cd[i], w, level=i + 1, take=data.shape[0]))

    return np.transpose(np.concatenate((np.array(rec_a), np.array(rec_d)), axis=0))


def wavelets(x: np.ndarray, wavelet_names: List[str] = None):
    if wavelet_names is None:
        wavelet_names = ['db2']

    wavelet_features = []
    for w in wavelet_names:
        wavelet_features.append(__wavelet_decomposition(x, w))

    return np.concatenate(wavelet_features, axis=0)


def spectral_residual(x: np.ndarray):
    pass


def spectrogram(signal: np.ndarray):
    pass


def power_spectral_density(signal: np.ndarray):
    pass
