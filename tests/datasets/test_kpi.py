"""
@Time    : 2021/10/26 11:50
@File    : test_kpi.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import KPIDataset


def test_kpidataset():
    dataset = KPIDataset(root='tests/data/kpi', download=False)
