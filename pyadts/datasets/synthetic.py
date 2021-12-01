"""
@Time    : 2021/10/22 11:31
@File    : synthetic.py
@Software: PyCharm
@Desc    : 
"""

from pyadts.generic import TimeSeriesDataset


class SyntheticDataset(TimeSeriesDataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
