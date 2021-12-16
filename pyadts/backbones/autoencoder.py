import torch
import torch.distributions as dist
import torch.nn as nn


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

        return x_dist, z_dist, z


class AAE(nn.Module):
    def __init__(self):
        super(AAE, self).__init__()

    def forward(self, x):
        pass
