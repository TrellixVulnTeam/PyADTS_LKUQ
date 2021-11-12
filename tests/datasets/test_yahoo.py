"""
@Time    : 2021/11/12 11:41
@File    : test_yahoo.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import YahooDataset


def test_yahoo_dataset():
    dataset = YahooDataset(root='tests/data/yahoo', category='A1Benchmark')
    dataset = YahooDataset(root='tests/data/yahoo', category='A4Benchmark')
