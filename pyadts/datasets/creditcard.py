"""
@Time    : 2021/10/28 2:02
@File    : creditcard.py
@Software: PyCharm
@Desc    : 
"""
import warnings
from pathlib import Path
from typing import Union

import pandas as pd

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import download_link, check_existence


class CreditCardDataset(TimeSeriesDataset):
    __link = 'https://www.openml.org/data/get_csv/1673544/phpKo8OWT'
    __filename = 'creditcard.csv'
    __md5 = '63aa311d4b9b32872b35f073ea9c8c2d'

    def __init__(self, root: str = None, download: bool = False):

        if root is None:
            root_path = Path.home() / 'creditcard'
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
                assert self.__check_integrity(root_path)
        else:
            assert self.__check_integrity(root_path)

        df = pd.read_csv(root_path / self.__filename)
        df.sort_values(by='Time', inplace=True)

        labels = df['Class'].map(lambda x: int(x.strip("'"))).values
        timestamps = df['Time'].values
        df.drop(columns=['Class', 'Time'], inplace=True)
        data = df.values

        super(CreditCardDataset, self).__init__(data=data, labels=labels, timestamps=timestamps)

    def __check_integrity(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

        if not check_existence(root / self.__filename, self.__md5):
            return False

        return True
