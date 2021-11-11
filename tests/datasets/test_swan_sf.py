"""
@Time    : 2021/10/29 0:06
@File    : test_swan_sf.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import SWANSFDataset


def test_swan_sfdataset():
    dataset = SWANSFDataset(root='tests/data/swansf', download=True)
