import numpy as np
import pandas as pd

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


def series_normalize(data_df: pd.DataFrame, meta_df: pd.DataFrame = None, method: str = 'minmax'):
    """
    Normalize given time series.

    Parameters
    ----------
    data_df : pd.DataFrame
        The data DataFrame.
    meta_df : pd.DataFrame, optional
        The meta DataFrame. default: `None`
    method : str
        The normalization method to use.
    """
    mask = None
    if meta_df is not None:
        assert 'missing' in meta_df.columns, 'To ignore missing values, you should pass the `meta_df` object!'
        mask = meta_df['missing'].values

    for column in data_df.columns:
        if method == 'minmax':
            data_df[column] = __normalize_minmax(data_df[column].values, mask=mask)
        elif method == 'negpos1':
            data_df[column] = __normalize_negpos1(data_df[column].values, mask=mask)
        elif method == 'zscore':
            data_df[column] = __normalize_zscore(data_df[column].values, mask=mask)
        else:
            raise ValueError('Invalid normalization method!')
