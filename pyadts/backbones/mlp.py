import torch
import torch.nn as nn


class DenseEncoder(nn.Module):
    def __init__(self, input_size: int, feature_dim: int, hidden_size: int, num_layers: int = 3,
                 use_batchnorm: bool = False, dropout: float = 0.0):
        super(DenseEncoder, self).__init__()

        layers = []
        assert num_layers >= 2

        layers.append(nn.Linear(input_size, hidden_size))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size, feature_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)

        return out


class DenseDecoder(nn.Module):
    def __init__(self, input_size: int, feature_dim: int, hidden_size: int, num_layers: int = 3,
                 use_batchnorm: bool = False, dropout: float = 0.0):
        super(DenseDecoder, self).__init__()

        layers = []
        assert num_layers >= 2

        layers.append(nn.Linear(feature_dim, hidden_size))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size, input_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        out = self.layers(z)

        return out
