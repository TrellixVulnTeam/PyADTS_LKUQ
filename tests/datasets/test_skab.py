"""
@Time    : 2021/10/26 18:23
@File    : test_skab.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import SKABDataset


def test_skabdataset():
    dataset = SKABDataset(root='tests/data/skab', category='other')
