"""
@Time    : 2021/11/10 19:43
@File    : test_gecco.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import GECCODataset


def test_geccodataset():
    dataset = GECCODataset(root='tests/data/gecco', download=True)
    print(dataset)
    print(dataset.data().shape, dataset.data().dtype)
    print(dataset.targets().shape, dataset.targets().dtype)
