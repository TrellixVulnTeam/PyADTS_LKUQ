import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset

from ..preprocessing import series_normalization, series_imputation
from ...utils.visualization import plot
from ..utils import timestamp_to_datetime


def from_pandas(df: pd.DataFrame, feature_columns: Union[List, str] = None, label_column: str = 'label',
                timestamp_column: str = 'timestamp'):
    if feature_columns is None:
        feature_columns = df.columns.tolist()
        if label_column is not None:
            feature_columns.remove(label_column)
        if timestamp_column is not None:
            feature_columns.remove(timestamp_column)

    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]

    return Series(value=df[feature_columns].values,
                  label=df[label_column].values if label_column is not None else None,
                  timestamp=df[timestamp_column].values if timestamp_column is not None else None)


def from_csv(file_path):
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)
    return from_pandas(df)


class Series(object):
    def __init__(self, value: np.ndarray, label: np.ndarray = None, timestamp: Union[np.ndarray, pd.DatetimeIndex] = None,
                 normalized=False, rearranged=False, imputed=False):
        self.__value = value
        self.__label = label
        self.__timestamp = timestamp

        if isinstance(self.__timestamp, np.ndarray):
            pass
        elif isinstance(self.__timestamp, pd.DatetimeIndex):
            pass
        else:
            raise ValueError('{}'.format(type(self.__timestamp)))

        self.__normalized = normalized
        self.__rearranged = rearranged
        self.__imputed = imputed

    # Following private methods
    def __check_shape(self):
        if self.__timestamp.shape != self.__value.shape:
            raise ValueError('Invalid shape of timestamp!')
        if self.__label.shape != self.__value.shape:
            raise ValueError('Invalid shape of label!')

    def __update_index(self, new_index):
        self.__value = self.__value[new_index]
        self.__timestamp = self.__timestamp[new_index]
        self.__label = self.__label[new_index]
        self.__missing = self.__missing[new_index]

    def __rearrange_index(self):
        # Sort
        sorted_index = np.argsort(self.__timestamp)
        self.__update_index(sorted_index)

        # Remove duplicated points
        unique_index = np.unique(self.__timestamp, return_index=True)[1]
        self.__update_index(unique_index)

        # Compute the minimum interval
        intervals = np.diff(self.__timestamp)
        min_interval = np.min(intervals)
        if min_interval <= 0:
            raise ValueError('Duplicated timestamp detected!')

        # All the time intervals should be multipliers of the minimum time interval
        for interval in intervals:
            if interval % min_interval != 0:
                raise ValueError('Misunderstanding `time_stamp` intervals!')

        # Reconstruct attributes
        reconstructed_index = (self.__timestamp - self.__timestamp[0]) // min_interval
        reconstructed_timestamp = np.arange(self.__timestamp[0], self.__timestamp[-1] + min_interval, min_interval,
                                            dtype=np.int64)

        assert reconstructed_timestamp[-1] == self.__timestamp[-1] and reconstructed_timestamp[0] == self.__timestamp[0]
        assert np.min(np.diff(reconstructed_timestamp)) == min_interval

        self.__missing = np.ones_like(reconstructed_timestamp, dtype=np.int)
        self.__missing[reconstructed_index] = 0

        reconstructed_length = (reconstructed_timestamp[-1] - reconstructed_timestamp[0]) // min_interval + 1
        assert len(reconstructed_timestamp) == reconstructed_length

        reconstructed_value = np.zeros(reconstructed_length, dtype=self.__value.dtype)
        reconstructed_value[reconstructed_index] = self.__value
        reconstructed_label = np.zeros(reconstructed_length, dtype=self.__label.dtype)
        reconstructed_label[reconstructed_index] = self.__label

        self.__value, self.__timestamp, self.__label = reconstructed_value, reconstructed_timestamp, reconstructed_label

        self.__check_shape()

    # Following properties
    @property
    def value(self):
        return self.__value

    @property
    def label(self):
        return self.__label

    @property
    def timestamp(self):
        return self.__timestamp

    @property
    def shape(self):
        return self.__value.shape

    @property
    def len(self):
        return self.__value.shape[0]

    @property
    def dim(self):
        return self.__value.shape[1]

    @property
    def normalized(self):
        return self.__normalized

    @property
    def rearrange(self):
        return self.__rearranged

    @property
    def imputed(self):
        return self.__imputed

    # Following statics
    @property
    def missing_num(self):
        if not self.__rearranged:
            warnings.warn('')
            return 0
        else:
            pass


    @property
    def missing_rate(self):
        pass

    @property
    def anomaly_num(self):
        if self.__label is None:
            warnings.warn('Anomaly labels are not provided, return zero.')
            return 0
        else:
            return np.count_nonzero(self.__label)

    @property
    def anomaly_rate(self):
        if self.__label is None:
            warnings.warn('Anomaly labels are not provided, return zero.')
            return 0.0
        else:
            return np.count_nonzero(self.__label)/self.__value.shape[0]

    # Visualization
    def plot(self):
        fig = plot(self.__value, self.__label, self.__timestamp)
        fig.show()

    # Following pre-processing
    def timestamp_rearrange(self):
        pass

    def normalize(self):
        pass

    def impute_missing(self):
        pass

    def split(self, ratios: Union[Tuple, List]):
        assert sum(ratios) == 1.0, 'The sum of ratios must be 1!'
        pass

    # Tensor dataset
    def to_tensor_dataset(self, return_label=True):
        if return_label:
            return TensorDataset(torch.from_numpy(self.__value.astype(np.float32)),
                                 torch.from_numpy(self.__label.astype(np.long)))
        else:
            return TensorDataset(torch.from_numpy(self.__value.astype(np.float32)))
