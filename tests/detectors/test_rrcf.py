"""
@Time    : 2021/12/9 9:16
@File    : test_rrcf.py
@Software: PyCharm
@Desc    : 
"""

from pyadts.datasets import MSLDataset
from pyadts.detectors import RRCF


def test_rrcf():
    model = RRCF()

    train_dataset = MSLDataset(root='tests/data/msl', train=True, download=False)
    test_dataset = MSLDataset(root='tests/data/msl', train=False, download=False)

    model.fit(train_dataset)

    scores = model.score(test_dataset)
