"""
@Time    : 2021/11/10 19:43
@File    : test_gecco.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import GECCODataset


def test_geccodataset():
    dataset = GECCODataset(root='tests/data/gecco', download=True)
