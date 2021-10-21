"""
@Time    : 2021/10/18 0:45
@File    : preprocessing.py
@Software: PyCharm
@Desc    : 
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def series_normalize(data_df: pd.DataFrame, method: str = 'minmax', inplace: bool = False):
    """
    Normalize given time series.

    Parameters
    ----------
    data_df : pd.DataFrame
        The data DataFrame.
    method : str
        The normalization method to use. choices: [`minmax`, `zscore`, `negpos1`]. default: `minmax`.
        - `minmax`: Scaling each series to the range (0, 1).
        - `zscore`: Scaling each series by zscore: :math:`z = \frac{x - u}{s}` :.
        - `negpos1`: Scaling each series to the range (-1, 1).
    """
    if np.isnan(data_df.values).any():
        warnings.warn('`NaN` detected, which will be ignored during normalization.')

    if not inplace:
        data_df = copy.deepcopy(data_df)

    if method == 'minmax':
        data_df.iloc[:, :] = MinMaxScaler().fit_transform(data_df.iloc[:, :].values)
    elif method == 'zscore':
        data_df.iloc[:, :] = StandardScaler().fit_transform(data_df.iloc[:, :].values)
    elif method == 'negpos1':
        data_df.iloc[:, :] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data_df.iloc[:, :].values)
    elif method == 'none':
        pass  # Do nothing
    else:
        raise ValueError('Invalid normalization method!')

    return data_df


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


def series_rearrange(data_df: pd.DataFrame, meta_df: pd.DataFrame, fill_label: float = 0, verbose: bool = True,
                     inplace: bool = False):
    """Time Series Rearrangement

    Parameters
    ----------
    old_data_df :
    old_meta_df :
    fill_label :
    verbose :

    Returns
    -------

    """
    assert (data_df.index == meta_df.index).all(), 'The indexes of `data_df` and `meta_df` dose not match!'

    if verbose:
        print('[INFO] Before processing, the shape of data: {}.'.format(data_df.shape))

    if inplace:
        old_data_df = data_df
        old_meta_df = meta_df
    else:
        old_data_df = copy.deepcopy(data_df)
        old_meta_df = copy.deepcopy(meta_df)

    # Drop duplicated rows
    old_meta_df.drop_duplicates(subset=['timestamp'], inplace=True, keep='first')
    old_data_df = old_data_df.loc[old_meta_df.index]

    # Sort rows
    old_data_df.sort_index(inplace=True)
    old_meta_df.sort_index(inplace=True)

    old_index = old_data_df.index
    datetime_series = old_index.to_series()
    timedelta_series = datetime_series.diff()[1:]
    min_interval = timedelta_series.min()

    if not (np.unique(timedelta_series % min_interval).astype(np.float) == np.zeros(1, dtype=np.float)).all():
        raise ValueError('Misunderstanding `time_stamp` intervals!')

    new_index = pd.date_range(start=datetime_series[0], end=datetime_series[-1], freq=min_interval, closed=None)
    new_data_df = pd.DataFrame(
        {column: np.full(shape=new_index.shape[0], fill_value=np.nan) for column in old_data_df.columns},
        index=new_index)
    new_data_df.loc[old_index] = old_data_df

    new_meta_df = pd.DataFrame(index=new_index)
    for column in old_meta_df.columns:
        if column == 'timestamp':
            new_meta_df['timestamp'] = new_index.to_series().apply(datetime.timestamp).astype(np.int64)
        elif column == 'label':
            new_meta_df['label'] = np.full(shape=new_index.shape[0], fill_value=fill_label)
            new_meta_df.loc[old_index, 'label'] = old_meta_df.loc[:, 'label']
        else:
            warnings.warn('Can not recognize attribute %s, ignored.' % column)

    if verbose:
        print('[INFO] Detected minimum interval: {}.'.format(min_interval))
        print('[INFO] After processing, the shape of data: {}.'.format(old_data_df.shape))

    if inplace:
        data_df = new_data_df
        meta_df = new_meta_df
        return data_df, meta_df
    else:
        return new_data_df, new_meta_df
