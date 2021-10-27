"""
@Time    : 2021/10/26 0:03
@File    : skab.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.std import tqdm

from pyadts.generic import TimeSeriesRepository


class SKABDataset(TimeSeriesRepository):
    __splits = ['valve1', 'valve2', 'other']
    __feature_columns = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature',
                         'Thermocouple', 'Voltage', 'Volume Flow RateRMS']

    def __init__(self, root: str, download: bool = False):
        super(SKABDataset, self).__init__()

        self.data = []
        self.labels = []

        root_path = Path(root)
        for split in self.__splits:
            data_files = list((root_path / split).glob('*.csv'))
            for file_path in tqdm(data_files, desc=f'::LOADING {split.upper()} DATA::'):
                df = pd.read_csv(file_path, delimiter=';')

                value = df.loc[:, self.__feature_columns].values.transpose()
                timestamp = df['datetime'].values
                anomaly = df['anomaly'].values.astype(int)
                change_point = df['changepoint'].values.astype(int)
                label = np.logical_or(anomaly, change_point).astype(int)

                self.data.append(value)
                self.labels.append(label)

        self.sep_indicators = np.cumsum([item.shape[-1] for item in self.data])
        self.data = np.concatenate(self.data, axis=-1)
        self.labels = np.concatenate(self.labels)
