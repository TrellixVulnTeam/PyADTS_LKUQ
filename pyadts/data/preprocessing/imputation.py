import copy

import pandas as pd


def series_impute(data_df: pd.DataFrame, method: str = 'zero', inplace: bool = False):
    """ Impute the series according given method. The `missing` attribute in `meta_df` is needed.

    Parameters
    ----------
    data_df : pd.DataFrame
        The data DataFrame
    method : str
        The imputation method, can be one of `['zero', 'linear`]`.
        - `zero`: Simply filling missing values with zeros.
        - `bfill`: Filling missing values with next valid observations
        - `ffill`: Filling missing values with last valid observations
        - `linear`: Filling missing values using linear interpolation
    Returns
    -------

    """
    if inplace:
        if method == 'zero':
            data_df.fillna(value=0, method=None, inplace=True)
        elif method == 'bfill':
            data_df.fillna(method='bfill', inplace=True)
        elif method == 'ffill':
            data_df.fillna(method='ffill', inplace=True)
        elif method == 'linear':
            data_df.interpolate(method='linear', inplace=True)
        elif method == 'none':
            pass  # Do nothing
        else:
            raise ValueError('Invalid imputation method!')

        return data_df
    else:
        if method == 'zero':
            new_data_df = data_df.fillna(value=0, method=None, inplace=False)
        elif method == 'bfill':
            new_data_df = data_df.fillna(method='bfill', inplace=False)
        elif method == 'ffill':
            new_data_df = data_df.fillna(method='ffill', inplace=False)
        elif method == 'linear':
            new_data_df = data_df.interpolate(method='linear', inplace=False)
        elif method == 'none':
            new_data_df = data_df  # Do nothing
        else:
            raise ValueError('Invalid imputation method!')

        return new_data_df
