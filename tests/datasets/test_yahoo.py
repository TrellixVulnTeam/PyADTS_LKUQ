"""
@Time    : 2021/11/12 11:41
@File    : test_yahoo.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import YahooDataset


def test_yahoo_dataset():
    for subset in ['A1Benchmark', 'A2Benchmark', 'A3Benchmark', 'A4Benchmark']:
        dataset = YahooDataset(root='tests/data/yahoo', subset=subset, download=False)
        print(dataset)
        print(dataset.data().shape, dataset.data().dtype)
        print(dataset.targets().shape, dataset.targets().dtype)
