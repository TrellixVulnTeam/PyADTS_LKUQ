from typing import List

import numpy as np
import pywt

from ...utils.scaffold_algorithms import spectral_residual_transform


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


def get_sr_feature(value):
    return spectral_residual_transform(value)


def get_wavelet_feature(value: np.ndarray, wavelet_names: List[str] = ['db2']):
    wavelet_features = []
    for w in wavelet_names:
        wavelet_features.append(__wavelet_decomposition(value, w))

    return np.concatenate(wavelet_features, axis=1)
