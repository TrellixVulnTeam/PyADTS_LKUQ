"""
@Time    : 2021/10/29 0:12
@File    : gecco.py
@Software: PyCharm
@Desc    : 
"""

from pyadts.generic import TimeSeriesRepository


class GECCODataset(TimeSeriesRepository):
    link = 'https://ndownloader.figshare.com/articles/12451142/versions/1'
    file_name = 'gecco.zip'

    def __init__(self, root: str, download: bool = False):
        super(GECCODataset, self).__init__()
