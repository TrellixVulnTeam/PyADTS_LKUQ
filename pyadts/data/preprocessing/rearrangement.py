import warnings
from datetime import datetime

import numpy as np
import pandas as pd


# from ..utils import timestamp_to_datetime


# def __rearrange_index(value: np.ndarray=None, label: np.ndarray=None, timestamp: np.ndarray=None, datetime: pd.Series=None):
#     # Sort
#     sorted_index = np.argsort(timestamp)
#
#     # Update index
#     value_sorted = value[sorted_index]
#     timestamp_sorted = timestamp[sorted_index]
#     datetime_sorted = datetime[sorted_index]
#     if label is not None:
#         label_sorted = label[sorted_index]
#
#     # Remove duplicated points
#     unique_unique = np.unique(timestamp_sorted, return_index=True)[1]
#     value_unique = value_sorted[sorted_index]
#     timestamp_unique = timestamp_sorted[sorted_index]
#     datetime_unique = datetime_sorted[sorted_index]
#     if label is not None:
#         label_unique = label_sorted[sorted_index]
#
#     # Compute the minimum interval
#     intervals = np.diff(timestamp_unique)
#     min_interval = np.min(intervals)
#     if min_interval <= 0:
#         raise ValueError('Duplicated timestamp detected!')
#
#     # All the time intervals should be multipliers of the minimum time interval
#     for interval in intervals:
#         if interval % min_interval != 0:
#             raise ValueError('Misunderstanding `time_stamp` intervals!')
#
#     # Reconstruct attributes
#     reconstructed_index = (timestamp_unique - timestamp_unique[0]) // min_interval
#     reconstructed_timestamp = np.arange(timestamp_unique[0], timestamp_unique[-1] + min_interval, min_interval,
#                                         dtype=np.int64)
#
#     assert reconstructed_timestamp[-1] == timestamp_unique[-1] and reconstructed_timestamp[0] == timestamp_unique[0]
#     assert np.min(np.diff(reconstructed_timestamp)) == min_interval
#
#     missing = np.ones_like(reconstructed_timestamp, dtype=np.int)
#     missing[reconstructed_index] = 0
#
#     reconstructed_length = (reconstructed_timestamp[-1] - reconstructed_timestamp[0]) // min_interval + 1
#     assert len(reconstructed_timestamp) == reconstructed_length
#
#     reconstructed_value = np.zeros(reconstructed_length, dtype=value_unique.dtype)
#     reconstructed_value[reconstructed_index] = value_unique
#     if label is not None:
#         reconstructed_label = np.zeros(reconstructed_length, dtype=label_unique.dtype)
#         reconstructed_label[reconstructed_index] = label_unique
#     reconstructed_datetime = pd.Series(reconstructed_timestamp).apply(timestamp_to_datetime)
#
#     return {'value': reconstructed_value, 'label': reconstructed_label if label is not None else None,
#             'timestamp': reconstructed_timestamp, 'datetime': reconstructed_datetime, 'missing': missing}


def series_rearrange(data_df: pd.DataFrame, meta_df: pd.DataFrame, fill_label: float = 0, verbose: bool = True):
    assert (data_df.index == meta_df.index).all(), 'The indexes of `data_df` and `meta_df` dose not match!'

    if verbose:
        print('[INFO] Before processing, the shape of data: {}.'.format(data_df.shape))

    # Drop duplicated rows
    meta_df.drop_duplicates(subset=['timestamp'], inplace=True, keep='first')
    data_df = data_df.loc[meta_df.index]

    # Sort rows
    data_df.sort_index(inplace=True)
    meta_df.sort_index(inplace=True)

    old_index = data_df.index
    datetime_series = old_index.to_series()
    timedelta_series = datetime_series.diff()[1:]
    min_interval = timedelta_series.min()

    if not (np.unique(timedelta_series % min_interval).astype(np.float) == np.zeros(1, dtype=np.float)).all():
        raise ValueError('Misunderstanding `time_stamp` intervals!')

    new_index = pd.date_range(start=datetime_series[0], end=datetime_series[-1], freq=min_interval, closed=None)
    new_data_df = pd.DataFrame(
        {column: np.full(shape=new_index.shape[0], fill_value=np.nan) for column in data_df.columns}, index=new_index)
    new_data_df.loc[old_index] = data_df

    new_meta_df = pd.DataFrame(index=new_index)
    for column in meta_df.columns:
        if column == 'timestamp':
            new_meta_df['timestamp'] = new_index.to_series().apply(datetime.timestamp).astype(np.int64)
        elif column == 'label':
            new_meta_df['label'] = np.full(shape=new_index.shape[0], fill_value=fill_label)
            new_meta_df.loc[old_index, 'label'] = meta_df.loc[:, 'label']
        else:
            warnings.warn('Can not recognize attribute %s, ignored.' % column)

    if verbose:
        print('[INFO] Detected minimum interval: {}.'.format(min_interval))
        print('[INFO] After processing, the shape of data: {}.'.format(data_df.shape))

    return new_data_df, new_meta_df
