from typing import Union, List

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, feature_dim: int, num_class: int, hidden_dims: Union[int, List[int]] = None,
                 use_batch_norm: bool = False,
                 dropout: float = 0.0):
        """

        Args:
            feature_dim ():
            num_class ():
            hidden_dims ():
            use_batch_norm ():
            dropout ():
        """
        super(MLP, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        if hidden_dims is None:
            self.layers = nn.Sequential(
                nn.Linear(feature_dim, num_class)
            )
        else:
            layers = [nn.Linear(feature_dim, hidden_dims[0])]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            for i in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dims[i]))
                layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Linear(hidden_dims[-1], num_class))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
