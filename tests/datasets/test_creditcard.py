"""
@Time    : 2021/10/28 2:05
@File    : test_creditcard.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import CreditCardDataset


def test_credit_card_dataset():
    dataset = CreditCardDataset(root='tests/data/creditcard', download=True)
    print(dataset)
    print(dataset.to_numpy().shape, dataset.to_numpy().dtype)
    print(dataset.targets().shape, dataset.targets().dtype)
