from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def timestamp_to_datetime(ts: Union[int, float]) -> datetime:
    return datetime.fromtimestamp(ts if isinstance(ts, int) else int(ts))


def datetime_to_timestamp(dt: datetime) -> int:
    return int(dt.timestamp())


def __missing_num(missing):
    return np.count_nonzero(missing)


def __missing_rate(missing):
    return np.count_nonzero(missing) / missing.shape[0]


def __anomaly_num(label):
    return np.count_nonzero(label)


def __anomaly_rate(label):
    return np.count_nonzero(label) / label.shape[0]


def dataset_statics(missing: np.ndarray, label: np.ndarray):
    return {'Missing num': __missing_num(missing), 'Missing rate': __missing_rate(missing),
            'Anomaly num': __anomaly_num(label), 'Anomaly rate': __anomaly_rate(label)}


def train_test_split(data_df: pd.DataFrame, meta_df: pd.DataFrame, train_ratio: float):
    train_num = int(data_df.shape[0]*train_ratio)
    return data_df.iloc[:train_num, :], meta_df.iloc[:train_num, :], data_df.iloc[train_num:, :], meta_df.iloc[train_num:, :]


def label_sampling(self, rate=1.0, method='segment'):
    rate = float(rate)
    assert 0.0 <= rate <= 1.0

    if method == 'segment':
        if rate == 1.0:
            return self
        elif rate == 0.0:
            return Series(value=self.value, timestamp=self.timestamp, label=None, name=self.name,
                          normalized=self.normalized)
        else:
            anomalies_num = np.count_nonzero(self.label) * rate
            sampled_label = np.copy(self.label).astype(np.int)
            start = np.where(np.diff(sampled_label) == 1)[0] + 1  # Heads of anomaly segments
            if sampled_label[0] == 1:
                start = np.concatenate([[0], start])
            end = np.where(np.diff(sampled_label) == -1)[0] + 1  # Tails of anomaly segments
            if sampled_label[-1] == 1:
                end = np.concatenate([end, [len(sampled_label)]])

            segments = np.arange(len(start))  # Segment ids
            np.random.shuffle(segments)

            # Iterate segments
            for i in range(len(start)):
                idx = (np.where(segments == i)[0]).item()
                sampled_label[start[idx]:end[idx]] = 0
                if np.count_nonzero(sampled_label) <= anomalies_num:
                    break

            return Series(value=self.value, timestamp=self.timestamp, label=sampled_label, name=self.name,
                          normalized=self.normalized)
    elif method == 'point':
        if rate == 1.0:
            return self
        elif rate == 0.0:
            return Series(value=self.value, timestamp=self.timestamp, label=None, name=self.name,
                          normalized=self.normalized)
        else:
            anomaly_indices = np.arange(self.length)[self.label == 1]
            selected_indices = np.random.choice(anomaly_indices,
                                                size=int(np.floor(anomaly_indices.shape[0] * (1 - rate))),
                                                replace=False)
            sampled_label = np.copy(self.label).astype(np.int)
            sampled_label[selected_indices] = 0

            return Series(value=self.value, timestamp=self.timestamp, label=sampled_label, name=self.name,
                          normalized=self.normalized)
    else:
        raise ValueError('Invalid label sampling method!')


def sliding_window():
    # TODO
    pass


def to_tensor_dataset(X: np.ndarray, y: np.ndarray = None) -> Dataset:
    pass
