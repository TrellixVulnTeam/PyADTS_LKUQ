"""
@Time    : 2021/10/27 1:54
@File    : test_msl.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import MSLDataset


def test_msldataset():
    dataset = MSLDataset(root='tests/data/msl', train=True, download=True)
    print(dataset)
    print(dataset.data().shape, dataset.data().dtype)
    print(dataset.targets().shape, dataset.targets().dtype)

    dataset = MSLDataset(root='tests/data/msl', train=False, download=True)
    print(dataset)
    print(dataset.data().shape, dataset.data().dtype)
    print(dataset.targets().shape, dataset.targets().dtype)
