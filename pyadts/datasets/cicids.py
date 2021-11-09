"""
@Time    : 2021/10/29 0:14
@File    : cicids.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path
from typing import Union

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import download_link, decompress_file, check_existence


class CICIDSDataset(TimeSeriesDataset):
    __link = 'http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip'
    __filename = 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    __md5 = '13e1c70d2b380bf5d90f82e60e7befb1'

    def __init__(self, root: str, download: bool = False):
        super(CICIDSDataset, self).__init__()
        root_path = Path(root)

        if download:
            if self.__check_integrity(root_path):
                print('Files are already downloaded and verified.')
            else:
                download_link(self.__link, root_path / self.__filename)
                decompress_file(root_path / self.__filename, root_path)
        else:
            self.__check_integrity(root_path)

    def __check_integrity(self, root: Union[str, Path]):
        if not check_existence(root / self.__filename, self.__md5):
            return False

        return True
