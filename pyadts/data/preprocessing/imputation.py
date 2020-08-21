import pandas as pd


def series_impute(data_df: pd.DataFrame, method: str = 'zero'):
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
    if method == 'zero':
        data_df.fillna(value=0, method=None, inplace=True)
    elif method == 'bfill':
        data_df.fillna(method='bfill', inplace=True)
    elif method == 'ffill':
        data_df.fillna(method='ffill', inplace=True)
    elif method == 'linear':
        data_df.interpolate(method='linear', inplace=True)
    else:
        raise ValueError('Invalid imputation method!')
