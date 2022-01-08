"""
@Time    : 2021/10/24 11:25
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import abc
from pathlib import Path
from typing import List, Tuple, Union, Iterable

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from prettytable import PrettyTable

from pyadts.utils.visualization import plot_series


class TimeSeriesDataset(abc.ABC):
    def __init__(self, data_list: Union[np.ndarray, List[np.ndarray]] = None,
                 label_list: Union[np.ndarray, List[np.ndarray]] = None,
                 timestamp_list: Union[np.ndarray, List[np.ndarray]] = None,
                 anomaly_score_list: Union[np.ndarray, List[np.ndarray]] = None,
                 prediction_list: Union[np.ndarray, List[np.ndarray]] = None,
                 dfs: List[pd.DataFrame] = None):
        """
        The abstract class of a time-series dataset. The dataset is  assumed to contain multiple
            multi-channel time-series

        Args:
            data_list:
            label_list:
            timestamp_list:
            anomaly_score_list:
            dfs:
        """
        if dfs is not None:
            self.dfs = dfs
        else:
            assert data_list is not None

            if isinstance(data_list, np.ndarray):
                data_list = [data_list]
            if label_list is not None and isinstance(label_list, np.ndarray):
                label_list = [label_list]
            if timestamp_list is not None and isinstance(timestamp_list, np.ndarray):
                timestamp_list = [timestamp_list]
            if anomaly_score_list is not None and isinstance(anomaly_score_list, np.ndarray):
                anomaly_score_list = [anomaly_score_list]
            if prediction_list is not None and isinstance(prediction_list, np.ndarray):
                prediction_list = [prediction_list]

            # TODO: change format to dict?
            # TODO: dealing with various timestamp formats
            # TODO: dealing with various shapes of `data`, `labels` and etc.

            self.dfs = []
            for idx, data_item in enumerate(data_list):
                df = pd.DataFrame(data_item, columns=[f'value-{i}' for i in range(data_item.shape[-1])])
                if label_list is not None:
                    assert idx < len(label_list)
                    assert len(data_item) == len(label_list[idx].reshape(-1))
                    df['__label'] = label_list[idx].reshape(-1).astype(np.long)
                else:
                    df['__label'] = np.full(len(data_item), fill_value=np.nan)

                if timestamp_list is not None:
                    assert idx < len(timestamp_list)
                    assert len(data_item) == len(timestamp_list[idx].reshape(-1))
                    df['__timestamp'] = timestamp_list[idx].reshape(-1).astype(np.int64)
                else:
                    df['__timestamp'] = np.arange(len(data_item), dtype=np.int64)

                if anomaly_score_list is not None:
                    assert idx < len(anomaly_score_list)
                    assert len(data_item) == len(anomaly_score_list[idx].reshape(-1))
                    df['__anomaly_score'] = anomaly_score_list[idx].reshape(-1)
                else:
                    df['__anomaly_score'] = np.full(len(data_item), fill_value=np.nan)

                if prediction_list is not None:
                    assert idx < len(prediction_list)
                    assert len(data_item) == len(prediction_list[idx].reshape(-1))
                    df['__prediction'] = prediction_list[idx].reshape(-1)
                else:
                    df['__prediction'] = np.full(len(data_item), fill_value=np.nan)

                self.dfs.append(df)

    # def detect(self, method, *args, **kwargs):
    #     model = method(*args, **kwargs)
    #     model.fit(self)
    #     return model.score(self)

    def plot(self, series_id: Union[int, List, Tuple[int, int]] = None,
             channel_id: Union[int, List, Tuple[int, int]] = None, show_ground_truth: bool = False,
             show_prediction: bool = False, style: Union[str, List[str]] = None, title: str = None,
             fig_size: Tuple[int, int] = None):
        if isinstance(series_id, int):
            series_id = [series_id]
        if isinstance(series_id, tuple):
            assert len(series_id) == 2
            series_id = list(range(*series_id))

        figs = []

        for i in series_id:
            if isinstance(channel_id, int):
                vis_data = self.values[i][:, channel_id: channel_id + 1]
            elif isinstance(channel_id, list):
                vis_data = self.values[i][:, channel_id]
            elif isinstance(channel_id, tuple):
                assert len(channel_id) == 2
                vis_data = self.values[i][:, channel_id[0]: channel_id[1]]
            else:
                raise ValueError

            if show_prediction and np.sum(np.isnan(self.predictions[i])) > 0:
                raise ValueError('Predictions not set or corrupted!')

            fig = plot_series(vis_data, timestamps=self.timestamps[i],
                              labels=self.labels[i] if show_ground_truth else None,
                              predictions=self.predictions[i] if show_prediction else None, style=style, title=title,
                              fig_size=fig_size)

            figs.append(fig)

        if len(figs) == 0:
            raise ValueError
        elif len(figs) == 1:
            return figs[0]
        else:
            return figs

    def to_numpy(self, window_size: int = None, stride: int = 1, return_labels: bool = False) -> Union[
        np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        data_list = []
        for df in self.dfs:
            current_data = df.loc[:, list(filter(lambda col: not col.startswith('__'), df.columns))].values
            if window_size is None:
                data_list.append(current_data)
            else:
                assert window_size <= len(current_data)
                tmp_data = []
                for i in range(0, len(current_data), stride):
                    if i + window_size > len(current_data):
                        break
                    tmp_data.append(current_data[i: i + window_size])
                tmp_data = np.stack(tmp_data, axis=0)
                data_list.append(tmp_data)

        data_list = np.concatenate(data_list, axis=0)

        if return_labels:
            label_list = []
            for df in self.dfs:
                current_label = df.loc[:, '__label'].values
                if window_size is None:
                    label_list.append(current_label)
                else:
                    tmp_label = []
                    for i in range(0, len(current_label), stride):
                        if i + window_size > len(current_label):
                            break
                        tmp_label.append(current_label[i: i + window_size])
                    tmp_label = np.stack(tmp_label, axis=0)
                    label_list.append(tmp_label)
            label_list = np.concatenate(label_list, axis=0)

            return data_list, label_list
        else:
            return data_list

    def to_tensor(self, window_size: int = None, stride: int = None, return_labels: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if return_labels:
            data, labels = self.to_numpy(window_size=window_size, stride=stride, return_labels=return_labels)
            return torch.from_numpy(data.astype(np.float32)), torch.from_numpy(labels.astype(np.long))
        else:
            data = self.to_numpy(window_size=window_size, stride=stride, return_labels=return_labels)
            return torch.from_numpy(data.astype(np.float32))

    # def targets(self, return_format: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
    #     label_list = [df.loc[:, '__label'].values for df in self.dfs]
    #     label_concat = np.concatenate(label_list, axis=0)
    #
    #     if return_format == 'numpy':
    #         return label_concat
    #     elif return_format == 'tensor':
    #         return torch.from_numpy(label_concat.astype(np.long))
    #     else:
    #         raise ValueError
    #
    # def windowed_data(self, window_size: int, stride: int = 1, return_format: str = 'numpy') -> Union[
    #     np.ndarray, torch.Tensor]:
    #     pass
    #
    # def windowed_targets(self, window_size: int, stride: int = 1, return_format: str = 'numpy') -> Union[
    #     np.ndarray, torch.Tensor]:
    #     pass

    def set_anomaly_score(self, anomaly_scores: List[np.ndarray]):
        for i, df in enumerate(self.dfs):
            assert len(anomaly_scores[i]) == df.shape[0]
            df['__anomaly_score'] = anomaly_scores[i]

    def set_prediction(self, predictions: List[np.ndarray]):
        for i, df in enumerate(self.dfs):
            assert len(predictions[i]) == df.shape[0]
            df['__prediction'] = predictions[i]

    @staticmethod
    def from_folder(root: Union[str, Path], suffix: str = '.csv', data_attributes: Iterable[str] = None,
                    timestamp_attribute: str = None, label_attribute: str = None):
        """
        Create a ``TimeSeriesDataset`` instance from a folder contains data files.

        Args:
            root:
            suffix:
            data_attributes:
            timestamp_attribute:
            label_attribute:

        Returns:

        """
        if isinstance(root, str):
            root = Path(root)

        data_list = []
        label_list = []
        timestamp_list = []

        file_list = root.glob(f'*{suffix}')
        for a_file in file_list:
            if a_file.name.endswith('.csv'):
                data = pd.read_csv(a_file)
                attributes = data.columns()
                data = data.to_dict()
            elif a_file.name.endswith('.npz'):
                data = np.load(str(a_file.absolute()))
                attributes = [key for key in data.keys() if not key.startswith('__')]
            elif a_file.name.endswith('.mat'):
                data = sio.loadmat(str(a_file.absolute()))
                attributes = [key for key in data.keys() if not key.startswith('__')]
            else:
                raise ValueError(f'Unsupported file format `{a_file.suffix}`!')

            if timestamp_attribute is not None:
                attributes.remove(timestamp_attribute)
                timestamp_list.append(data[timestamp_attribute])

            if label_attribute is not None:
                attributes.remove(label_attribute)
                label_list.append(data[label_attribute])

            if data_attributes is not None:
                attributes = list(data_attributes)

            data_list.append(np.stack([data[attr] for attr in attributes], axis=-1))

        if len(label_list) == 0:
            label_list = None
        if len(timestamp_list) == 0:
            timestamp_list = None

        dataset = TimeSeriesDataset(data_list=data_list, label_list=label_list, timestamp_list=timestamp_list)
        return dataset

    @staticmethod
    def from_iterable(dfs: Iterable[pd.DataFrame]):
        """
        Create a ``TimeSeriesDataset`` instance from iterable dataframes.

        Args:
            dfs:

        Returns:

        """
        return TimeSeriesDataset(dfs=list(dfs))

    @property
    def values(self) -> List[np.ndarray]:
        return [
            df.loc[:, [col for col in df.columns if not col.startswith('__')]].values for df in self.dfs
        ]

    @property
    def timestamps(self) -> List[np.ndarray]:
        return [
            df.loc[:, '__timestamp'].values for df in self.dfs
        ]

    @property
    def labels(self) -> List[np.ndarray]:
        return [
            df.loc[:, '__label'].values for df in self.dfs
        ]

    @property
    def scores(self) -> List[np.ndarray]:
        return [
            df.loc[:, '__anomaly_score'].values for df in self.dfs
        ]

    @property
    def predictions(self) -> List[np.ndarray]:
        return [
            df.loc[:, '__prediction'].values for df in self.dfs
        ]

    @property
    def shape(self):
        return self.to_numpy().shape

    @property
    def num_series(self):
        return len(self.dfs)

    @property
    def num_points(self):
        return sum([len(df) for df in self.dfs])

    @property
    def num_channels(self):
        return len(list(filter(lambda col: not col.startswith('__'), self.dfs[0].columns)))

    def __getitem__(self, item):
        return self.dfs[item]

    def __len__(self):
        return len(self.dfs)

    def __repr__(self):
        table = PrettyTable()
        table.align = 'c'
        table.valign = 'm'
        table.field_names = ['ID', '# Channels', '# Points']

        for i, df in enumerate(self.dfs):
            table.add_row(
                [f'series-{i}', len(list(filter(lambda col: not col.startswith('__'), df.columns))), df.shape[0]])

        return table.get_string()
