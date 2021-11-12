"""
@Time    : 2021/10/27 1:54
@File    : test_smd.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import SMDDataset


def test_smddataset():
    dataset = SMDDataset(root='tests/data/smd', download=False)
