"""
@Time    : 2021/10/27 1:54
@File    : test_smap.py
@Software: PyCharm
@Desc    : 
"""
import os
import shutil

from pyadts.datasets import SMAPDataset


def test_smapdataset():
    if os.path.exists('tests/data/SMAP'):
        shutil.rmtree('tests/data/SMAP')
    os.makedirs('tests/data/SMAP')

    dataset = SMAPDataset(root='tests/data/SMAP', download=True)
