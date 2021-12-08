"""
@Time    : 2021/10/24 11:25
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import abc
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable


class TimeSeriesDataset(abc.ABC):
    def __init__(self, data_list: Union[np.ndarray, List[np.ndarray]],
                 label_list: Union[np.ndarray, List[np.ndarray]] = None,
                 timestamp_list: Union[np.ndarray, List[np.ndarray]] = None,
                 anomaly_score_list: Union[np.ndarray, List[np.ndarray]] = None):
        """
        The abstract class of a time-series dataset. The dataset is  assumed to contain multiple
            multi-channel time-series

        Args:
            data_list ():
            label_list ():
            timestamp_list ():
            anomaly_score_list ():
        """
        if isinstance(data_list, np.ndarray):
            data_list = [data_list]
        if label_list is not None and isinstance(label_list, np.ndarray):
            label_list = [label_list]
        if timestamp_list is not None and isinstance(timestamp_list, np.ndarray):
            timestamp_list = [timestamp_list]
        if anomaly_score_list is not None and isinstance(anomaly_score_list, np.ndarray):
            anomaly_score_list = [anomaly_score_list]

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
                df['__label'] = np.zeros(len(data_item))

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
                df['__anomaly_score'] = np.arange(len(data_item))

            self.dfs.append(df)

    def plot(self, show: bool = True):
        # fig = plot_series(np.transpose(self.dfs[0].loc[:, list(filter(lambda col: not col.startswith('__'), self.dfs[0].columns))].values))
        # if show:
        #     fig.show()
        pass

    def data(self, format: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        data_list = [df.loc[:, list(filter(lambda col: not col.startswith('__'), df.columns))].values for df in
                     self.dfs]
        data_concat = np.concatenate(data_list, axis=0)

        if format == 'numpy':
            return data_concat
        elif format == 'tensor':
            return torch.from_numpy(data_concat.astype(np.float32))
        else:
            raise ValueError

    def targets(self, format: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        label_list = [df.loc[:, '__label'].values for df in self.dfs]
        label_concat = np.concatenate(label_list, axis=0)

        if format == 'numpy':
            return label_concat
        elif format == 'tensor':
            return torch.from_numpy(label_concat.astype(np.long))
        else:
            raise ValueError

    # def windowed_data(self, window_size: int, stride: int = 1, format: str='numpy') -> Union[np.ndarray, torch.Tensor]:
    #     data_list = []
    #     for df in self.dfs:
    #         data_item = df.loc[:, list(filter(lambda col: not col.startswith('__'), df.columns))].values
    #         print(data_item.shape)
    #         data_item_window = sliding_window(data_item.transpose(), window_size=window_size, stride=stride)
    #         data_item_window = np.swapaxes(data_item_window, 0, 1)
    #         data_list.append(data_item_window)
    #
    #     data_concat = np.concatenate(data_list, axis=0)
    #
    #     if format == 'numpy':
    #         return data_concat
    #     elif format == 'tensor':
    #         return torch.from_numpy(data_concat.astype(np.float32))
    #     else:
    #         raise ValueError
    #
    # def windowed_targets(self, window_size: int, stride: int = 1, format: str='numpy') -> Union[np.ndarray, torch.Tensor]:
    #     label_list = []
    #     for df in self.dfs:
    #         label_item = df.loc[:, '__label'].values
    #         label_item_window = sliding_window(label_item, window_size=window_size, stride=stride)
    #         label_list.append(label_item_window)
    #
    #     label_concat = np.concatenate(label_list, axis=0)
    #
    #     if format == 'numpy':
    #         return label_concat
    #     elif format == 'tensor':
    #         return torch.from_numpy(label_concat.astype(np.float32))
    #     else:
    #         raise ValueError

    def __repr__(self):
        table = PrettyTable()
        table.align = 'c'
        table.field_names = ['ID', '# Channels', '# Points']

        for i, df in enumerate(self.dfs):
            table.add_row(
                [f'series-{i}', len(list(filter(lambda col: not col.startswith('__'), df.columns))), df.shape[0]])

        return table.get_string()
