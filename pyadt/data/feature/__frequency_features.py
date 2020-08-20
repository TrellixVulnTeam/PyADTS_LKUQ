from typing import List

import numpy as np

EPS=1e-8
MAG_WINDOW=3


def __average_filter(values, n=3):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res


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
    trans = np.fft.fft(value)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
    eps_index = np.where(mag <= EPS)[0]
    mag[eps_index] = EPS

    mag_log = np.log(mag)
    mag_log[eps_index] = 0

    spectral = np.exp(mag_log - __average_filter(mag_log, n=MAG_WINDOW))

    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)

    return mag


def get_wavelet_feature(value: np.ndarray, wavelet_names: List[str] = ['db2']):
    wavelet_features = []
    for w in wavelet_names:
        wavelet_features.append(__wavelet_decomposition(value, w))

    return np.concatenate(wavelet_features, axis=1)
