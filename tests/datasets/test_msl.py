"""
@Time    : 2021/10/27 1:54
@File    : test_msl.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import MSLDataset


def test_msldataset():
    dataset = MSLDataset(root='tests/data/msl', download=True)
