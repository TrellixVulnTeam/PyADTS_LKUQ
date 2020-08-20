import warnings

import numpy as np
import pandas as pd


def series_impute(data_df: pd.DataFrame, meta_df: pd.DataFrame, method: str = 'zero'):
    """ Impute the series according given method. The `missing` attribute in `meta_df` is needed.

    Parameters
    ----------
    data_df : pd.DataFrame
        The data DataFrame
    meta_df : pd.DataFrame
        The meta DataFrame
    method : str
        The imputation method, can be one of `['zero', 'linear`]`.
        - `zero`: Simply filling missing values with zeros.
        - `linear`:
    Returns
    -------

    """
    assert 'missing' in meta_df.columns, 'The `missing` attribute is not found. ' \
                                         'Please run `pyadts.data.preprocessing.series_rearrange` ' \
                                         'before invoking this function.'

    if np.count_nonzero(meta_df['missing'].values) == 0:
        warnings.warn('The series contains no missing values, skipped.')
        return

    if method == 'zero':
        data_df['value'].fillna(0, inplace=True)
    elif method == 'linear':
        data_df['value'].interpolate(method='linear', inplace=True)
    else:
        raise ValueError('Invalid imputation method!')
