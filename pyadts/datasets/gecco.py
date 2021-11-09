"""
@Time    : 2021/10/29 0:12
@File    : gecco.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path
from typing import Union

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import download_link, check_existence


class GECCODataset(TimeSeriesDataset):
    __link = 'https://zenodo.org/record/3884398/files/1_gecco2018_water_quality.csv?download=1'
    __filename = '1_gecco2018_water_quality.csv'

    def __init__(self, root: str, download: bool = False):
        super(GECCODataset, self).__init__()
        root_path = Path(root)

        if download:
            if self.__check_integrity(root_path):
                print('Files are already downloaded and verified.')
            else:
                download_link(self.__link, root_path / self.__filename)
        else:
            self.__check_integrity(root_path)

    def __check_integrity(self, root: Union[str, Path]):
        if not check_existence(root / self.__filename):
            return False

        return True
