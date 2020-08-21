import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append('..')

from pyadts.data.preprocessing import series_rearrange


def test_series_rearrange():
    index = pd.date_range(start='2018/1/1', end='2018/3/1', freq='1min')
    x = np.linspace(-10, 10, index.shape[0])
    value = 2 * np.sin(x) + np.cos(x) + np.random.randn(x.shape[0])
    value2 = np.sin(x) + 0.5 * np.cos(x + 1) + np.random.randn(x.shape[0])
    timestamps = index.to_series().apply(datetime.timestamp).astype(np.int)

    missing = np.random.choice(np.arange(1, index.shape[0] - 1), size=int(index.shape[0] * 0.9) - 2, replace=False)
    missing = np.concatenate([np.array([0]), missing, np.array([index.shape[0] - 1])])
    missing = np.sort(missing)
    index_missing = index[missing]
    value_missing = value[missing]
    value2_missing = value2[missing]
    timestamps_missing = timestamps[missing]

    labels = np.zeros_like(value_missing, dtype=np.int)
    anomalies = np.random.choice(np.arange(labels.shape[0]), size=int(labels.shape[0] * 0.1), replace=False)
    labels[anomalies] = 1
    value_missing[anomalies] += np.random.randn(anomalies.shape[0]) * np.mean(value_missing)
    value2_missing[anomalies] += np.random.randn(anomalies.shape[0]) * np.mean(value2_missing)

    shuffle = np.roll(np.arange(value_missing.shape[0]), int(value_missing.shape[0] * 0.05))
    index_shuffled = index_missing[shuffle]
    value_shuffled = value_missing[shuffle]
    value2_shuffled = value2_missing[shuffle]
    timestamps_shuffled = timestamps_missing[shuffle]
    labels_shuffled = labels[shuffle]

    data_df = pd.DataFrame({'value': value_shuffled, 'value2': value2_shuffled}, index=index_shuffled)
    meta_df = pd.DataFrame({'timestamp': timestamps_shuffled, 'label': labels_shuffled}, index=index_shuffled)

    # print(data_df)
    # print(meta_df)

    data_df, meta_df = series_rearrange(data_df, meta_df)

    assert data_df.index.shape == index.shape, '{} - {}'.format(data_df.index.shape, index.shape)
    assert (data_df.index == index).all(), '{}'.format(data_df.index[data_df.index != index])
    assert data_df.index.shape == meta_df.index.shape, '{} - {}'.format(data_df.index.shape, meta_df.index.shape)
    assert (data_df.index == meta_df.index).all(), '{}'.format(data_df.index[data_df.index != meta_df.index])
    assert np.isnan(data_df.values).sum() == (index.shape[0] - missing.shape[0]) * 2, '{}'.format(
        np.isna(data_df.values).sum())
