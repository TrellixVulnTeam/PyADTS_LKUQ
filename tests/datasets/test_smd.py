"""
@Time    : 2021/10/27 1:54
@File    : test_smd.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import SMDDataset


def test_smddataset():
    dataset = SMDDataset(root='tests/data/smd', train=True, download=False)
    print(dataset)
    print(dataset.to_numpy().shape, dataset.to_numpy().dtype)
    print(dataset.targets().shape, dataset.targets().dtype)

    dataset = SMDDataset(root='tests/data/smd', train=False, download=False)
    print(dataset)
    print(dataset.to_numpy().shape, dataset.to_numpy().dtype)
    print(dataset.targets().shape, dataset.targets().dtype)
