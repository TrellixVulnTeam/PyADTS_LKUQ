"""
@Time    : 2021/11/9 0:51
@File    : test_cicids.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import CICIDSDataset


def test_cicidsdataset():
    dataset = CICIDSDataset(root='tests/data/cicids', download=True)
