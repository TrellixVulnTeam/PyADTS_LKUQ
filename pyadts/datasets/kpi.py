"""
@Time    : 2021/10/28 2:02
@File    : creditcard.py
@Software: PyCharm
@Desc    :
"""
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tqdm.std import tqdm

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.data import rearrange_dataframe
from pyadts.utils.io import check_existence


class KPIDataset(TimeSeriesDataset):
    __splits = {
        'first': 'phase2_train.csv',
        'second': 'phase2_ground_truth.hdf'
    }
    __file_list = {
        'phase2_train.csv': '787967d365157bc2228f1153ba32334d',
        'phase2_ground_truth.hdf': '5c4e834ce210eea4e9f755ca045806ec'
    }

    def __init__(self, root: str = None, download: bool = False):
        if root is None:
            root_path = Path.home() / 'kpi'
            warnings.warn(
                f'The `root` path of the dataset is not set, using user home dir {str(root_path)} as default.')
        else:
            root_path = Path(root)

        if download:
            raise ValueError('The KPI dataset should be downloaded manually. '
                             'Please download the dataset at `http://iops.ai/dataset_detail/?id=7`!')
        else:
            self.__check_integrity(root_path)

        first_df = pd.read_csv(root_path / self.__splits['first'])
        second_df = pd.read_hdf(root_path / self.__splits['second'])
        df = pd.concat([first_df, second_df])

        kpi_ids = np.unique(df['KPI ID'].values.astype(str))
        df_group_by_id = {kpi: df[df['KPI ID'] == kpi] for kpi in kpi_ids}

        data = []
        labels = []
        timestamps = []

        for key, df in tqdm(df_group_by_id.items(), desc='::LOADING DATA::', colour='cyan'):
            df = rearrange_dataframe(df.drop(columns=['KPI ID']), time_col='timestamp', sort_by_time=True,
                                     resampling=True, tackle_missing='fzero')

            data.append(df['value'].values.reshape(-1, 1))
            labels.append(df['label'].values.reshape(-1))
            timestamps.append(df['timestamp'].values.reshape(-1))

        super(KPIDataset, self).__init__(data_list=data, label_list=labels, timestamp_list=timestamps)

    def __check_integrity(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

        for key, value in self.__file_list.items():
            if not check_existence(root / key, value):
                return False

        return True
