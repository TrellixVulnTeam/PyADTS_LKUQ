"""
@Time    : 2021/10/27 1:54
@File    : test_smap.py
@Software: PyCharm
@Desc    : 
"""

from pyadts.datasets import SMAPDataset


def test_smapdataset():
    dataset = SMAPDataset(root='tests/data/smap', train=True, download=True)
    print(dataset)
    print(dataset.to_numpy().shape, dataset.to_numpy().dtype)
    print(dataset.targets().shape, dataset.targets().dtype)

    dataset = SMAPDataset(root='tests/data/smap', train=False, download=True)
    print(dataset)
    print(dataset.to_numpy().shape, dataset.to_numpy().dtype)
    print(dataset.targets().shape, dataset.targets().dtype)
