from typing import List, Union

import numpy as np
import pandas as pd

from .__decomposition_features import get_stl_feature
from .__frequency_features import get_sr_feature, get_wavelet_feature
from .__regression_features import get_simplees_feature, get_holt_feature, get_addes_feature, get_sarima_feature
from .__simple_features import get_log_feature, get_diff_feature, get_diff2_feature
from .__window_features import get_window_feature


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
