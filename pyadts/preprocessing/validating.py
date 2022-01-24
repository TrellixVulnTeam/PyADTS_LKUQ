import numpy as np

from pyadts.generic import TimeSeriesDataset


def rearrange_timestamps(data: TimeSeriesDataset) -> TimeSeriesDataset:
    """
    original timeseries may contain non-monotonic increasing, replicated and missing timestamps

    Args:
        data:

    Returns:

    """
    new_values = []
    new_labels = []
    new_timestamps = []

    for df in data.dfs:
        df.sort_values(by='__timestamp', inplace=True)
        timestamp = df['__timestamp'].values
        intervals = np.diff(timestamp)
        if len(intervals) == 1:
            new_value = df.loc[:, data.data_attributes].values
            new_timestamp = df.loc[:, '__timestamp'].vlaues
            new_label = df.loc[:, '__label'].values
        else:
            min_interval = np.min(intervals)
            assert (np.unique(np.diff(timestamp)) % min_interval == 0).all()
            new_timestamp = np.arange(timestamp[0], timestamp[-1] + 1, min_interval)
            assert new_timestamp[-1] == timestamp[-1]
            new_value = np.full(shape=(len(new_timestamp), data.num_channels), fill_value=np.nan)
            new_label = np.zeros(len(new_timestamp))
            indicator = np.in1d(new_timestamp, timestamp)

            new_value[indicator] = df.loc[:, data.data_attributes].values
            new_label[indicator] = df.loc[:, '__label'].values

        new_values.append(new_value)
        new_labels.append(new_label)
        new_timestamps.append(new_timestamp)

    return TimeSeriesDataset(new_values, new_labels, new_timestamps)


def fill_missing(data: TimeSeriesDataset, fill_method: str = 'bfill') -> TimeSeriesDataset:
    """
    bfill:
    0 | 3.5
    1 | NaN -> 4.5
    2 | NaN -> 4.5
    3 | 4.5

    ffill:
    0 | 3.5
    1 | NaN -> 3.5
    2 | 4.5

    Args:
        data:
        fill_method:

    Returns:

    """
    new_dfs = []
    for df in data.dfs:
        df[list(filter(lambda col: not col.startswith('__'), df.columns))] = df[
            list(filter(lambda col: not col.startswith('__'), df.columns))].fillna(method=fill_method)
        new_dfs.append(df)
    return TimeSeriesDataset.from_iterable(new_dfs)
