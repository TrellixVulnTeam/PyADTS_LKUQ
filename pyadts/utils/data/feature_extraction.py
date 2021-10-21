"""
@Time    : 2021/10/18 0:44
@File    : feature_extraction.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np
import pywt
from statsmodels.tsa.api import SARIMAX, ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import STL

from .__interpacf import interpolated_acf, dominant_period


def get_all_features(data_df: pd.DataFrame, get_sr: bool = True, get_simple: bool = True, get_sarima: bool = True,
                     get_addes: bool = True, get_holt: bool = True, get_simplees: bool = True, get_wavelet: bool = True,
                     get_stl: bool = True, get_window: bool = True, window_list: List[int] = None,
                     period: Union[int, str] = 'auto', fillna: bool = True) -> pd.DataFrame:
    if data_df.shape[1] != 1:
        raise ValueError('Only support 1-dimensional time series!')

    features = []
    value = data_df.iloc[:, 0].values

    if get_sr:
        print('[INFO] Processing spectral residual feature...')
        sr_feature = get_sr_feature(value)
        features.append(sr_feature)

    if get_simple:
        print('[INFO] Processing simple features...')
        log_feature = get_log_feature(value)
        diff_feature = get_diff_feature(value)
        diff2_feature = get_diff2_feature(value)
        features.append(log_feature)
        features.append(diff_feature)
        features.append(diff2_feature)

    if get_sarima:
        print('[INFO] Processing SARIMA features...')
        sarima_feature = get_sarima_feature(value)
        features.append(sarima_feature)

    if get_addes:
        print('[INFO] Processing ExponentialSmoothing features...')
        addes_feature = get_addes_feature(value)
        features.append(addes_feature)

    if get_holt:
        print('[INFO] Processing Holt features...')
        holt_feature = get_holt_feature(value)
        features.append(holt_feature)

    if get_simplees:
        print('[INFO] Processing SimpleExpSmoothing features...')
        simplees_feature = get_simplees_feature(value)
        features.append(simplees_feature)

    if get_wavelet:
        print('[INFO] Processing wavelet features...')
        wavelet_feature = get_wavelet_feature(value)
        features.append(wavelet_feature)

    if get_stl:
        print('[INFO] Processing STL features...')
        stl_feature = get_stl_feature(value, period=period)
        features.append(stl_feature)

    if get_window:
        assert window_list is not None
        print('[INFO] Processing window features...')
        for window in window_list:
            window_feature = get_window_feature(value, window_size=window)
            features.append(window_feature)

    min_length = np.min([feature.shape[0] for feature in features])

    for i, feature in enumerate(features):
        if feature.shape[0] > min_length:
            features[i] = features[i][-min_length:]

        if features[i].ndim == 1:
            features[i] = features[i].reshape(-1, 1)

        features[i] = np.transpose(features[i])

    features = np.transpose(np.concatenate(features))
    features = pd.DataFrame(features, index=data_df.index[-features.shape[0]:])
    if fillna:
        features.fillna(0, inplace=True)

    return features


def get_log_feature(value: np.ndarray) -> np.ndarray:
    return np.log(value)


def get_diff_feature(value: np.ndarray) -> np.ndarray:
    return np.diff(value, prepend=0)


def get_diff2_feature(value: np.ndarray) -> np.ndarray:
    return np.diff(np.diff(value, prepend=0), prepend=0)


def get_window_feature(value: np.ndarray, window_size: int, verbose: bool = True):
    assert window_size * 2 <= value.shape[0]

    start_point = 2 * window_size
    # start_accum = 0
    data = []

    progress_bar = tqdm(np.arange(start_point, len(value)), desc='SLIDING_WINDOW') if verbose else np.arange(
        start_point, len(value))

    for i in progress_bar:
        # the datum to put into the data pool
        datum = []

        # fill the datum with features related to windows
        mean_w = np.mean(value[i - window_size:i + 1])
        var_w = np.mean((np.asarray(value[i - window_size:i + 1]) - mean_w) ** 2)
        # var_w = np.var(time_series[i-k:i+1])

        mean_w_and_1 = mean_w + (value[i - window_size - 1] - value[i]) / (window_size + 1)
        var_w_and_1 = np.mean((np.asarray(value[i - window_size - 1:i]) - mean_w_and_1) ** 2)
        # mean_w_and_1 = np.mean(time_series[i-k-1:i])
        # var_w_and_1 = np.var(time_series[i-k-1:i])

        mean_2w = np.mean(value[i - 2 * window_size:i - window_size + 1])
        var_2w = np.mean((np.asarray(value[i - 2 * window_size:i - window_size + 1]) - mean_2w) ** 2)
        # var_2w = np.var(time_series[i-2*k:i-k+1])

        # diff of sliding windows
        diff_mean_1 = mean_w - mean_w_and_1
        diff_var_1 = var_w - var_w_and_1

        # diff of jumping windows
        diff_mean_w = mean_w - mean_2w
        diff_var_w = var_w - var_2w

        # f1
        datum.append(mean_w)  # [0:2] is [0,1]
        # f2
        datum.append(var_w)
        # f3
        datum.append(diff_mean_1)
        # f4
        datum.append(diff_mean_1 / (mean_w_and_1 + 1e-10))
        # f5
        datum.append(diff_var_1)
        # f6
        datum.append(diff_var_1 / (var_w_and_1 + 1e-10))
        # f7
        datum.append(diff_mean_w)
        # f8
        datum.append(diff_mean_w / (mean_2w + 1e-10))
        # f9
        datum.append(diff_var_w)
        # f10
        datum.append(diff_var_w / (var_2w + 1e-10))

        # diff of sliding/jumping windows and current value
        # f11
        datum.append(value[i] - mean_w_and_1)
        # f12
        datum.append(value[i] - mean_2w)

        data.append(np.asarray(datum))

    return np.asarray(data)


def __auto_period(value: np.ndarray):
    lag, acf = interpolated_acf(np.arange(value.shape[0]), value)
    period = dominant_period(lag, acf, plot=False)

    return int(period)


def get_stl_feature(value: np.ndarray, period: Union[int, str] = 'auto', seasonal: int = 7,
                    robust: bool = True) -> np.ndarray:
    if isinstance(period, str):
        assert period == 'auto'

    if period == 'auto':
        print('[INFO] Finding period automatically...')
        period = __auto_period(value)
        if np.isnan(period):
            raise ValueError('No period found!')

    stl = STL(value, seasonal=seasonal, period=period, robust=robust)
    res = stl.fit()
    feature = np.transpose(np.stack((res.trend, res.seasonal, res.resid)))

    return feature


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


def get_sarima_feature(value):
    predict = SARIMAX(value,
                      trend='n').fit(disp=0).get_prediction()
    return value - predict.predicted_mean


def get_addes_feature(value):
    predict = ExponentialSmoothing(value, trend='add').fit(smoothing_level=1)
    return value - predict.fittedvalues


def get_simplees_feature(value):
    predict = SimpleExpSmoothing(value).fit(smoothing_level=1)
    return value - predict.fittedvalues


def get_holt_feature(value):
    predict = Holt(value).fit(smoothing_level=1)
    return value - predict.fittedvalues
