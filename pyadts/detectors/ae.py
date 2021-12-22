"""
@Time    : 2021/10/25 15:13
@File    : ae.py
@Software: PyCharm
@Desc    : 
"""

from pyadts.backbones.mlp import MLP
from pyadts.generic import Detector, TimeSeriesDataset


class AutoencoderDetector(Detector):
    def __init__(self, hidden_dim: int, arch: str = 'mlp', device: str = 'cpu'):
        super(AutoencoderDetector, self).__init__()

        if arch == 'mlp':
            self.encoder = MLP()
            self.decoder = MLP()
        elif arch == 'conv':
            raise NotImplementedError
        else:
            raise ValueError

    def fit(self, x: TimeSeriesDataset, y=None):
        pass

    def predict(self, x: TimeSeriesDataset):
        pass

    def score(self, x: TimeSeriesDataset):
        pass
