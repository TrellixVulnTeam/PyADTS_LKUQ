"""
@Time    : 2021/10/27 1:54
@File    : test_smap.py
@Software: PyCharm
@Desc    : 
"""

from pyadts.datasets import SMAPDataset


def test_smapdataset():
    dataset = SMAPDataset(root='tests/data/smap', download=True)
