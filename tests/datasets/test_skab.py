"""
@Time    : 2021/10/26 18:23
@File    : test_skab.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import SKABDataset


def test_skabdataset():
    for subset in ['valve1', 'valve2', 'other']:
        dataset = SKABDataset(root='tests/data/skab', subset=subset, download=False)
        print(dataset)
        print(dataset.data().shape, dataset.data().dtype)
        print(dataset.targets().shape, dataset.targets().dtype)
