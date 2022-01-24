from typing import Callable

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

from pyadts.generic import TimeSeriesDataset


def __perform_scale(data: TimeSeriesDataset, copy: bool, scaler: Callable) -> TimeSeriesDataset:
    new_dfs = []
    for df in data.dfs:
        if copy:
            df = df.copy()
        df.loc[:, data.data_attributes] = scaler(df.loc[:, data.data_attributes].values)
        new_dfs.append(df)
    return TimeSeriesDataset.from_iterable(new_dfs)


def min_max_scale(data: TimeSeriesDataset, copy=False) -> TimeSeriesDataset:
    return __perform_scale(data, copy, MinMaxScaler().fit_transform)


def standard_scale(data: TimeSeriesDataset, copy=False) -> TimeSeriesDataset:
    return __perform_scale(data, copy, StandardScaler().fit_transform)


def robust_scale(data: TimeSeriesDataset, copy=False) -> TimeSeriesDataset:
    return __perform_scale(data, copy, RobustScaler().fit_transform)


def quantile_scale(data: TimeSeriesDataset, copy=False) -> TimeSeriesDataset:
    return __perform_scale(data, copy, QuantileTransformer().fit_transform)
