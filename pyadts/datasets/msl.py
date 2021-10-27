"""
@Time    : 2021/10/18 17:59
@File    : msl.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.std import tqdm

from pyadts.generic import TimeSeriesRepository


class MSLDataset(TimeSeriesRepository):
    __splits = ['train', 'test']
    __label_folder = 'test_label'

    def __init__(self, root: str):
        super(MSLDataset, self).__init__()

        self.data = []
        self.labels = []

        root_path = Path(root)
        for split in self.__splits:
            data_files = list((root_path / split).glob('*.csv'))
            for file_path in tqdm(data_files, desc=f'::LOADING {split.upper()} DATA::'):
                df = pd.read_csv(file_path)

                value = df.values.transpose()
                if split == 'test':
                    label = pd.read_csv(root_path / self.__label_folder / file_path.name).values[:, -1].astype(int)
                else:
                    label = np.zeros(value.shape[-1], dtype=int)

                self.data.append(value)
                self.labels.append(label)

        self.sep_indicators = np.cumsum([item.shape[-1] for item in self.data])
        self.data = np.concatenate(self.data, axis=-1)
        self.labels = np.concatenate(self.labels)
