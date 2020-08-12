import warnings

import numpy as np
import pandas as pd


def __filling_zeros(value: np.ndarray, missing: np.ndarray):
    new_value = np.zeros_like(value)
    new_value[missing == 0] = value

    return new_value


def __filling_linear(value: np.ndarray, missing: np.ndarray):
    new_value = pd.Series(value)
    new_value = new_value.interpolate(method='linear')

    return new_value.values


# def filling_pad(value: np.ndarray, missing: np.ndarray):
#     new_value = pd.Series(value)
#     new_value = new_value.interpolate(method='pad')
#
#     return new_value.values


def series_impute(value: np.ndarray, missing: np.ndarray, method: str = 'zero'):
    assert value is not None
    assert missing is not None

    if np.count_nonzero(missing) == 0:
        warnings.warn('The series contains no missing values, skipped.')
        return value

    if method == 'zero':
        return __filling_zeros(value, missing)
    elif method == 'linear':
        return __filling_linear(value, missing)
    else:
        raise ValueError('Invalid imputation method!')
