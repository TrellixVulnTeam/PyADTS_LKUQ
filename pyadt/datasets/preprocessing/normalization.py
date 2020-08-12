import warnings

import numpy as np

from ..utils import handle_zeros


def __normalize_minmax(value: np.ndarray, mask: np.ndarray=None):
    if mask is not None:
        value_excluded = value[np.logical_not(mask)]
    else:
        value_excluded = value

    min_value = np.min(value_excluded)
    max_value = np.max(value_excluded)

    return (value - min_value) / handle_zeros(max_value - min_value)


def __normalize_negpos1(value: np.ndarray, mask: np.ndarray=None):
    if mask is not None:
        value_excluded = value[np.logical_not(mask)]
    else:
        value_excluded = value

    min_value = np.min(value_excluded)
    max_value = np.max(value_excluded)

    return ((value - min_value) / handle_zeros(max_value - min_value) - 0.5) / 0.5


def __normalize_zscore(value: np.ndarray, mask: np.ndarray=None):
    if mask is not None:
        value_excluded = value[np.logical_not(mask)]
    else:
        value_excluded = value

    mean_value = np.mean(value_excluded)
    std_value = np.std(value_excluded)

    return (value - mean_value) / handle_zeros(std_value)


def series_normalize(value: np.ndarray, mask: np.ndarray=None, method: str = 'minmax'):

    if method == 'minmax':
        return __normalize_minmax(value, mask=mask)
    elif method == 'negpos1':
        return __normalize_negpos1(value, mask=mask)
    elif method == 'zscore':
        return __normalize_zscore(value, mask=mask)
    else:
        raise ValueError('Invalid normalization method!')
