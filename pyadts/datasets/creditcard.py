"""
@Time    : 2021/10/28 2:02
@File    : creditcard.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path
from typing import Union

import pandas as pd

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import download_link, check_existence


class CreditCardDataset(TimeSeriesDataset):
    __link = 'https://www.openml.org/data/get_csv/1673544/phpKo8OWT'
    __filename = 'creditcard.csv'
    __md5 = '63aa311d4b9b32872b35f073ea9c8c2d'

    def __init__(self, root: str, download: bool = False):
        super(CreditCardDataset, self).__init__()
        root_path = Path(root)

        if download:
            if self.__check_integrity(root_path):
                print('Files are already downloaded and verified.')
            else:
                download_link(self.__link, root_path / self.__filename)
        else:
            self.__check_integrity(root_path)

        df = pd.read_csv(root_path / self.__filename)
        print(df)

    def __check_integrity(self, root: Union[str, Path]):
        if not check_existence(root / self.__filename, self.__md5):
            return False

        return True
