import warnings
from typing import Union

import numpy as np

from torch.utils.data import Dataset, TensorDataset

from datetime import datetime


def timestamp_to_datetime(ts: Union[int, float]) -> datetime:
    return datetime.fromtimestamp(ts if isinstance(ts, int) else int(ts))


def datetime_to_timestamp(dt: datetime) -> int:
    return int(dt.timestamp())


def handle_zeros(value):
    EPS = 1e-6
    if value < EPS:
        warnings.warn(f'Zero denominator detected. Replace it by {EPS}.')
        return EPS
    return value


def __missing_num(missing):
    return np.count_nonzero(missing)


def __missing_rate(missing):
    return np.count_nonzero(missing) / missing.shape[0]


def __anomaly_num(label):
    return np.count_nonzero(label)


def __anomaly_rate(label):
    return np.count_nonzero(label)/label.shape[0]


def dataset_statics(missing: np.ndarray, label: np.ndarray):
    return {'Missing num': __missing_num(missing), 'Missing rate': __missing_rate(missing),
            'Anomaly num': __anomaly_num(label), 'Anomaly rate': __anomaly_rate(label)}


def train_test_split():
    # TODO
    pass


def label_sampling():
    # TODO
    pass


def sliding_window():
    # TODO
    pass


def to_tensor_dataset(X: np.ndarray, y: np.ndarray=None) -> Dataset:
    pass
