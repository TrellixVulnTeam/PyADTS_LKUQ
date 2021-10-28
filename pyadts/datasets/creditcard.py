"""
@Time    : 2021/10/28 2:02
@File    : creditcard.py
@Software: PyCharm
@Desc    : 
"""
import os

from pyadts.generic import TimeSeriesRepository
from pyadts.utils.io import download_link, check_existence


class CreditCardDataset(TimeSeriesRepository):
    link = 'https://www.openml.org/data/get_csv/1673544/phpKo8OWT'
    file_name = 'creditcard.csv'
    md5 = '63aa311d4b9b32872b35f073ea9c8c2d'

    def __init__(self, root: str, download: bool = False):
        super(CreditCardDataset, self).__init__()

        if download:
            if not check_existence(os.path.join(root, self.file_name), self.md5):
                download_link(self.link, os.path.join(root, self.file_name))
            else:
                print('Files are already downloaded and verified.')
