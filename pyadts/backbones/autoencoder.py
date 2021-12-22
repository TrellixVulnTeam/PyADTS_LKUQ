from typing import Type

import torch
import torch.distributions as dist
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder: Type[nn.Module], decoder: Type[nn.Module], input_size: int, feature_dim: int,
                 hidden_size: int, **kwargs):
        super(Autoencoder, self).__init__()

        self.encoder = encoder(input_size=input_size, feature_dim=feature_dim, hidden_size=hidden_size, **kwargs)
        self.decoder = decoder(input_size=input_size, feature_dim=feature_dim, hidden_size=hidden_size, **kwargs)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)

        return x_rec, z


class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder: Type[nn.Module], decoder: Type[nn.Module], input_size: int, feature_dim: int,
                 hidden_size: int, num_samples: int = 1, **kwargs):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = encoder(input_size=input_size, feature_dim=feature_dim * 2, hidden_size=hidden_size, **kwargs)
        self.decoder = decoder(input_size=input_size * 2, feature_dim=feature_dim, hidden_size=hidden_size, **kwargs)
        self.sigmoid = nn.Sigmoid()

        self.feature_dim = feature_dim
        self.num_samples = num_samples

    def forward(self, x):
        # batch_size = x.shape[0]
        out = self.encoder(x)
        z_mean, z_std = torch.chunk(out, chunks=2, dim=-1)
        z_std = self.sigmoid(z_std)
        if self.training:
            z_dist = dist.Normal(torch.zeros_like(z_mean), torch.ones_like(z_std))
            z_samples = z_dist.sample((self.num_samples,)) * z_std.unsqueeze(0) + z_mean.unsqueeze(0)
            # z = z.view(self.num_samples * batch_size, self.feature_dim)
            z = z_samples.mean(0)
        else:
            z_dist = dist.Normal(z_mean, z_std)
            z_samples = z_dist.sample((self.num_samples,))
            # z = z.view(self.num_samples * batch_size, self.feature_dim)
            z = z_samples.mean(0)

        x_rec = self.decoder(z)
        # x_rec = x_rec.view(self.num_samples, batch_size, -1)
        x_mean, x_std = torch.chunk(x_rec, chunks=2, dim=-1)
        x_std = self.sigmoid(x_std)
        x_dist = dist.Normal(x_mean, x_std)

        return x_dist, z_dist, z_samples


class ELBO(nn.Module):
    def __init__(self):
        super(ELBO, self).__init__()

        self.z_prior = None

    def forward(self, x: torch.Tensor, z: torch.Tensor, p_xz: dist.Distribution, q_zx: dist.Distribution):
        x = x.unsqueeze(0)
        log_p_xz = p_xz.log_prob(x)
        log_q_zx = torch.sum(q_zx.log_prob(z), -1)
        if self.z_prior is None:
            self.z_prior = dist.Normal(torch.zeros(z.shape[-1]).to(z.device), torch.ones(z.shape[-1]).to(z.device))
        log_p_z = torch.sum(self.z_prior.log_prob(z), -1)
        loss = - torch.mean(torch.sum(log_p_xz, -1) + log_p_z - log_q_zx)

        return loss


class AdversarialAutoencoder(nn.Module):
    def __init__(self):
        super(AdversarialAutoencoder, self).__init__()

    def forward(self, x):
        pass
