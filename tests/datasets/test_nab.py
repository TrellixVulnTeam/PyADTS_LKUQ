"""
@Time    : 2021/11/11 17:10
@File    : test_nab.py
@Software: PyCharm
@Desc    : 
"""
from pyadts.datasets import NABDataset


def test_nabdataset():
    for subset in ['artificialNoAnomaly', 'artificialWithAnomaly', 'realAdExchange', 'realAWSCloudwatch',
                   'realKnownCause', 'realTraffic', 'realTweets']:
        dataset = NABDataset(root='tests/data/nab', subset=subset, download=True)
        print(dataset)
        print(dataset.to_numpy().shape, dataset.to_numpy().dtype)
        print(dataset.targets().shape, dataset.targets().dtype)
