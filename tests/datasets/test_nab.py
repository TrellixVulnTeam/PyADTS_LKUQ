"""
@Time    : 2021/11/11 17:10
@File    : test_nab.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import NABDataset


def test_nabdataset():
    dataset = NABDataset(root='tests/data/nab', category='realTraffic', download=True)
