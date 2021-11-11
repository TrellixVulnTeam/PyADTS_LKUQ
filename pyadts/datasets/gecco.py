"""
@Time    : 2021/10/29 0:12
@File    : gecco.py
@Software: PyCharm
@Desc    : 
"""
import warnings
from pathlib import Path
from typing import Union

import pandas as pd

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import download_link, check_existence


class GECCODataset(TimeSeriesDataset):
    __link = 'https://zenodo.org/record/3884398/files/1_gecco2018_water_quality.csv?download=1'
    __filename = '1_gecco2018_water_quality.csv'

    def __init__(self, root: str = None, download: bool = False):
        super(GECCODataset, self).__init__()

        if root is None:
            root_path = Path.home() / 'gecco'
            warnings.warn(
                f'The `root` path of the dataset is not set, using user home dir {str(root_path)} as default.')
        else:
            root_path = Path(root)

        if download:
            if self.__check_integrity(root_path):
                print('Files are already downloaded and verified.')
            else:
                print('Start downloading...')
                download_link(self.__link, root_path / self.__filename)
                self.__check_integrity(root_path)
        else:
            self.__check_integrity(root_path)

        df = pd.read_csv(root_path / self.__filename, index_col=0)
        df.sort_values(by='Time', inplace=True)
        df['EVENT'] = df['EVENT'].map({False: 0, True: 1})
        df.drop(columns=['Time'])

        self.labels = df['EVENT'].values
        df.drop(columns=['EVENT'], inplace=True)
        self.data = df.values

    def __check_integrity(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

        if not check_existence(root / self.__filename):
            return False

        return True
