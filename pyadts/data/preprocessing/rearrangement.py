import numpy as np
import pandas as pd

from ..utils import timestamp_to_datetime


def __rearrange_index(value: np.ndarray=None, label: np.ndarray=None, timestamp: np.ndarray=None, datetime: pd.Series=None):
    # Sort
    sorted_index = np.argsort(timestamp)

    # Update index
    value_sorted = value[sorted_index]
    timestamp_sorted = timestamp[sorted_index]
    datetime_sorted = datetime[sorted_index]
    if label is not None:
        label_sorted = label[sorted_index]

    # Remove duplicated points
    unique_unique = np.unique(timestamp_sorted, return_index=True)[1]
    value_unique = value_sorted[sorted_index]
    timestamp_unique = timestamp_sorted[sorted_index]
    datetime_unique = datetime_sorted[sorted_index]
    if label is not None:
        label_unique = label_sorted[sorted_index]

    # Compute the minimum interval
    intervals = np.diff(timestamp_unique)
    min_interval = np.min(intervals)
    if min_interval <= 0:
        raise ValueError('Duplicated timestamp detected!')

    # All the time intervals should be multipliers of the minimum time interval
    for interval in intervals:
        if interval % min_interval != 0:
            raise ValueError('Misunderstanding `time_stamp` intervals!')

    # Reconstruct attributes
    reconstructed_index = (timestamp_unique - timestamp_unique[0]) // min_interval
    reconstructed_timestamp = np.arange(timestamp_unique[0], timestamp_unique[-1] + min_interval, min_interval,
                                        dtype=np.int64)

    assert reconstructed_timestamp[-1] == timestamp_unique[-1] and reconstructed_timestamp[0] == timestamp_unique[0]
    assert np.min(np.diff(reconstructed_timestamp)) == min_interval

    missing = np.ones_like(reconstructed_timestamp, dtype=np.int)
    missing[reconstructed_index] = 0

    reconstructed_length = (reconstructed_timestamp[-1] - reconstructed_timestamp[0]) // min_interval + 1
    assert len(reconstructed_timestamp) == reconstructed_length

    reconstructed_value = np.zeros(reconstructed_length, dtype=value_unique.dtype)
    reconstructed_value[reconstructed_index] = value_unique
    if label is not None:
        reconstructed_label = np.zeros(reconstructed_length, dtype=label_unique.dtype)
        reconstructed_label[reconstructed_index] = label_unique
    reconstructed_datetime = pd.Series(reconstructed_timestamp).apply(timestamp_to_datetime)

    return {'value': reconstructed_value, 'label': reconstructed_label if label is not None else None,
            'timestamp': reconstructed_timestamp, 'datetime': reconstructed_datetime, 'missing': missing}


def __rearrange_index(data_df: pd.DataFrame, meta_df: pd.DataFrame):
    assert (data_df.index == meta_df.index).all(), 'The indexes of `data_df` and `meta_df` dose not match!'

    data_df.sort_index(inplace=True)
    meta_df.sort_index(inplace=True)

    data_df.index.to_series()


def series_rearrange(value: np.ndarray=None, label: np.ndarray=None, timestamp: np.ndarray=None, datetime: pd.Series=None):
    assert value is not None
    assert timestamp is not None

    return __rearrange_index(value=value, label=label, timestamp=timestamp, datetime=datetime)
