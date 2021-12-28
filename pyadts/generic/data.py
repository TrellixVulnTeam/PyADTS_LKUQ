"""
@Time    : 2021/10/24 11:25
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import abc
from pathlib import Path
from typing import List, Tuple, Union, Iterable, Type

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from prettytable import PrettyTable

from pyadts.generic import Detector
from pyadts.utils.visualization import plot_series


class TimeSeriesDataset(abc.ABC):
    def __init__(self, data_list: Union[np.ndarray, List[np.ndarray]] = None,
                 label_list: Union[np.ndarray, List[np.ndarray]] = None,
                 timestamp_list: Union[np.ndarray, List[np.ndarray]] = None,
                 anomaly_score_list: Union[np.ndarray, List[np.ndarray]] = None,
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

            # TODO: change format to dict?
            # TODO: dealing with various timestamp formats
            # TODO: dealing with various shapes of `data`, `labels` and etc.

            self.dfs = []
            for idx, data_item in enumerate(data_list):
                df = pd.DataFrame(data_item, columns=[f'value-{i}' for i in range(data_item.shape[-1])])
                if label_list is not None:
                    assert idx < len(label_list)
                    assert len(data_item) == len(label_list[idx].reshape(-1))
                    df['__label'] = label_list[idx].reshape(-1)
                else:
                    df['__label'] = np.full(len(data_item), fill_value=np.nan)

                if timestamp_list is not None:
                    assert idx < len(timestamp_list)
                    assert len(data_item) == len(timestamp_list[idx].reshape(-1))
                    df['__timestamp'] = timestamp_list[idx].reshape(-1)
                else:
                    df['__timestamp'] = np.arange(len(data_item))

                if anomaly_score_list is not None:
                    assert idx < len(anomaly_score_list)
                    assert len(data_item) == len(anomaly_score_list[idx].reshape(-1))
                    df['__anomaly_score'] = anomaly_score_list[idx].reshape(-1)
                else:
                    df['__anomaly_score'] = np.full(len(data_item), fill_value=np.nan)

                self.dfs.append(df)

    def detect(self, method: Type[Detector], *args, **kwargs):
        model = method(*args, **kwargs)
        model.fit(self)
        return model.score(self)

    def plot(self, series_id: Union[int, List, Tuple[int, int]] = None,
             channel_id: Union[int, List, Tuple[int, int]] = None, show: bool = True):
        if isinstance(series_id, int):
            series_id = [series_id]

        for i in series_id:
            if isinstance(channel_id, int) or isinstance(channel_id, list):
                vis_data = self.values[i][:, channel_id]
            elif isinstance(channel_id, tuple):
                assert len(channel_id) == 2
                vis_data = self.values[i][:, channel_id[0]: channel_id[1]]
            else:
                raise ValueError

            fig = plot_series(self.values[i], )

            if show:
                fig.show()
            yield fig

    def to_numpy(self) -> np.ndarray:
        data_list = [df.loc[:, list(filter(lambda col: not col.startswith('__'), df.columns))].values for df in
                     self.dfs]
        data_concat = np.concatenate(data_list, axis=0)

        return data_concat

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.to_numpy()).float()

    def targets(self, return_format: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        label_list = [df.loc[:, '__label'].values for df in self.dfs]
        label_concat = np.concatenate(label_list, axis=0)

        if return_format == 'numpy':
            return label_concat
        elif return_format == 'tensor':
            return torch.from_numpy(label_concat.astype(np.long))
        else:
            raise ValueError

    def windowed_data(self, window_size: int, stride: int = 1, return_format: str = 'numpy') -> Union[
        np.ndarray, torch.Tensor]:
        pass

    def windowed_targets(self, window_size: int, stride: int = 1, return_format: str = 'numpy') -> Union[
        np.ndarray, torch.Tensor]:
        pass

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
        return self.dfs[0].shape[-1]

    def __getitem__(self, item):
        return self.dfs[item]

    def __repr__(self):
        table = PrettyTable()
        table.align = 'c'
        table.field_names = ['ID', '# Channels', '# Points']

        for i, df in enumerate(self.dfs):
            table.add_row(
                [f'series-{i}', len(list(filter(lambda col: not col.startswith('__'), df.columns))), df.shape[0]])

        return table.get_string()
