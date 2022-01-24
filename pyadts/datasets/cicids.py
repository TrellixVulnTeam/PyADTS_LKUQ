"""
@Time    : 2021/10/29 0:14
@File    : cicids.py
@Software: PyCharm
@Desc    : 
"""
import shutil
import warnings
from pathlib import Path
from typing import Union

import pandas as pd

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import download_link, decompress_file, check_existence


class CICIDSDataset(TimeSeriesDataset):
    __link = 'http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip'
    __filename = 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    __md5 = '13e1c70d2b380bf5d90f82e60e7befb1'

    def __init__(self, root: str = None, download: bool = False):

        if root is None:
            root_path = Path.home() / 'cicids'
            warnings.warn(
                f'The `root` path of the dataset is not set, using user home dir {str(root_path)} as default.')
        else:
            root_path = Path(root)

        if download:
            if self.__check_integrity(root_path):
                print('Files are already downloaded and verified.')
            else:
                if not check_existence(root_path / self.__filename):
                    print('Start downloading...')
                    download_link(self.__link, root_path / self.__filename)
                print('Start decompressing...')
                decompress_file(root_path / self.__filename, root_path)
                shutil.move(str(root_path / 'MachineLearningCVE' / self.__filename),
                            str(root_path / self.__filename))
                assert self.__check_integrity(root_path)
        else:
            assert self.__check_integrity(root_path)

        df = pd.read_csv(root_path / self.__filename, skipinitialspace=True)
        df.drop(columns=['Destination Port', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags'],
                inplace=True)  # drop non-numerical columns

        df['Label'] = df['Label'].map(
            {'BENIGN': 0, "Web Attack Brute Force": 1, "Web Attack Sql Injection": 1, "Web Attack XSS": 1})

        labels = df['Label'].values
        df.drop(columns=['Label'], inplace=True)
        data = df.values

        super(CICIDSDataset, self).__init__(data=data, labels=labels)

    def __check_integrity(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

        if not check_existence(root / self.__filename, self.__md5):
            return False

        return True
