"""
@Time    : 2021/10/29 0:05
@File    : swansf.py
@Software: PyCharm
@Desc    : 
"""
import os

from pyadts.generic import TimeSeriesRepository
from pyadts.utils.io import download_link, check_existence


class SWANSFDataset(TimeSeriesRepository):
    link = 'https://bitbucket.org/gsudmlab/mvtsdata_toolkit/downloads/petdataset_01.zip'
    file_name = 'petdataset_01.zip'

    # md5 = '63aa311d4b9b32872b35f073ea9c8c2d'

    def __init__(self, root: str, download: bool = False):
        super(SWANSFDataset, self).__init__()

        if download:
            if not check_existence(os.path.join(root, self.file_name), self.md5):
                download_link(self.link, os.path.join(root, self.file_name))
            else:
                print('Files are already downloaded and verified.')
