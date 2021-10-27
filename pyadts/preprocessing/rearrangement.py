"""
@Time    : 2021/10/18 11:10
@File    : rearrangement.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple

import numpy as np


def rearrange(x: np.ndarray, timestamp: np.ndarray, *arrays: Tuple[np.ndarray]):
    pass

# def series_rearrange(data_df: pd.DataFrame, meta_df: pd.DataFrame, fill_label: float = 0, verbose: bool = True,
#                      inplace: bool = False):
#     assert (data_df.index == meta_df.index).all(), 'The indexes of `data_df` and `meta_df` dose not match!'
#
#     if verbose:
#         print('[INFO] Before processing, the shape of data: {}.'.format(data_df.shape))
#
#     if inplace:
#         old_data_df = data_df
#         old_meta_df = meta_df
#     else:
#         old_data_df = copy.deepcopy(data_df)
#         old_meta_df = copy.deepcopy(meta_df)
#
#     # Drop duplicated rows
#     old_meta_df.drop_duplicates(subset=['timestamp'], inplace=True, keep='first')
#     old_data_df = old_data_df.loc[old_meta_df.index]
#
#     # Sort rows
#     old_data_df.sort_index(inplace=True)
#     old_meta_df.sort_index(inplace=True)
#
#     old_index = old_data_df.index
#     datetime_series = old_index.to_series()
#     timedelta_series = datetime_series.diff()[1:]
#     min_interval = timedelta_series.min()
#
#     if not (np.unique(timedelta_series % min_interval).astype(np.float) == np.zeros(1, dtype=np.float)).all():
#         raise ValueError('Misunderstanding `time_stamp` intervals!')
#
#     new_index = pd.date_range(start=datetime_series[0], end=datetime_series[-1], freq=min_interval, closed=None)
#     new_data_df = pd.DataFrame(
#         {column: np.full(shape=new_index.shape[0], fill_value=np.nan) for column in old_data_df.columns},
#         index=new_index)
#     new_data_df.loc[old_index] = old_data_df
#
#     new_meta_df = pd.DataFrame(index=new_index)
#     for column in old_meta_df.columns:
#         if column == 'timestamp':
#             new_meta_df['timestamp'] = new_index.to_series().apply(datetime.timestamp).astype(np.int64)
#         elif column == 'label':
#             new_meta_df['label'] = np.full(shape=new_index.shape[0], fill_value=fill_label)
#             new_meta_df.loc[old_index, 'label'] = old_meta_df.loc[:, 'label']
#         else:
#             warnings.warn('Can not recognize attribute %s, ignored.' % column)
#
#     if verbose:
#         print('[INFO] Detected minimum interval: {}.'.format(min_interval))
#         print('[INFO] After processing, the shape of data: {}.'.format(old_data_df.shape))
#
#     if inplace:
#         data_df = new_data_df
#         meta_df = new_meta_df
#         return data_df, meta_df
#     else:
#         return new_data_df, new_meta_df
