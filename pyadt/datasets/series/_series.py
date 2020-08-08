from typing import List, Union

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset


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

    return Series(feature=df[feature_columns].values,
                  label=df[label_column].values if label_column is not None else None,
                  timestamp=df[timestamp_column].values if timestamp_column is not None else None)


def from_csv(file_path):
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)
    return from_pandas(df)


class Series(object):
    def __init__(self, feature: np.ndarray, label: np.ndarray = None, timestamp: pd.DatetimeIndex = None):
        self.__feature = feature
        self.__label = label
        self.__timestamp = timestamp

    @property
    def feature(self):
        return self.__feature

    @property
    def label(self):
        return self.__label

    @property
    def timestamp(self):
        return self.__timestamp

    @property
    def len(self):
        return self.feature.shape[0]

    @property
    def dim(self):
        return self.feature.shape[1]

    def to_tensor_dataset(self, return_label=True):
        if return_label:
            return TensorDataset(torch.from_numpy(self.feature.astype(np.float32)),
                                 torch.from_numpy(self.label.astype(np.long)))
        else:
            return TensorDataset(torch.from_numpy(self.feature.astype(np.float32)))
