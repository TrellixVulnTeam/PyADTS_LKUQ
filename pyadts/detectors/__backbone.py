from typing import Union, List

import torch
import torch.distributions as dist
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, feature_dim: int, num_class: int, hidden_dim: Union[int, List[int]] = None, norm: bool = False,
                 dropout: float = 0.0):
        """

        Args:
            feature_dim ():
            num_class ():
            hidden_dim ():
            num_layers ():
            norm ():
            dropout ():
        """
        super(MLP, self).__init__()

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        if hidden_dim is None:
            self.layers = nn.Sequential(
                nn.Linear(feature_dim, num_class)
            )
        else:
            layers = []
            layers.append(nn.Linear(feature_dim, hidden_dim[0]))
            if norm:
                layers.append(nn.BatchNorm1d(hidden_dim[0]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            for i in range(1, len(hidden_dim)):
                layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                if norm:
                    layers.append(nn.BatchNorm1d(hidden_dim[i]))
                layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Linear(hidden_dim[-1], num_class))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Autoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, hidden_dim: int):
        super(Autoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = hidden_dim

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)

        return x_rec, z


class ELBO(nn.Module):
    def __init__(self):
        super(ELBO, self).__init__()

    def forward(self, x):
        pass


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, hidden_dim: int, num_samples: int = 1):
        super(VAE, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.encoder(x)
        z_mean, z_std = torch.chunk(out, chunks=2, dim=-1)
        if self.training:
            z_dist = dist.Normal(torch.zeros_like(z_mean), torch.ones_like(z_std))
            z = z_dist.sample((self.num_samples,)) * z_std.unsqueeze(0) + z_mean.unsqueeze(0)
            z = z.view(self.num_samples * batch_size, self.hidden_dim)
        else:
            z_dist = dist.Normal(z_mean, z_std)
            z = z_dist.sample((self.num_samples,))
            z = z.view(self.num_samples * batch_size, self.hidden_dim)

        x_rec = self.decoder(z)
        x_rec = x_rec.view(self.num_samples, batch_size, -1)
        x_mean, x_std = torch.chunk(x_rec, chunks=2, dim=-1)
        x_dist = dist.Normal(x_mean, x_std)
