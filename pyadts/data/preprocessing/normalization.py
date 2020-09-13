import copy

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
