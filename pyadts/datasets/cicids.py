"""
@Time    : 2021/10/29 0:14
@File    : cicids.py
@Software: PyCharm
@Desc    : 
"""

from pyadts.generic import TimeSeriesRepository


class CICIDSDataset(TimeSeriesRepository):
    link = 'http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip'
    file_name = 'MachineLearningCSV.zip'

    def __init__(self, root: str, download: bool = False):
        super(CICIDSDataset, self).__init__()
