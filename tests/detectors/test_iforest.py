"""
@Time    : 2021/12/9 22:36
@File    : test_iforest.py
@Software: PyCharm
@Desc    : 
"""

from pyadts.datasets import MSLDataset
from pyadts.detectors import IForest


def test_iforest():
    model = IForest()

    train_dataset = MSLDataset(root='tests/data/msl', train=True, download=False)
    test_dataset = MSLDataset(root='tests/data/msl', train=False, download=False)

    model.fit(train_dataset)

    scores = model.score(test_dataset)
    print(test_dataset.shape, scores.shape)
