import torch
import torch.distributions as dist
import torch.nn as nn


class ELBO(nn.Module):
    def __init__(self, hidden_dim: int):
        super(ELBO, self).__init__()

        self.hidden_dim = hidden_dim
        self.z_prior = dist.Normal(torch.zeros(hidden_dim), torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor, z: torch.Tensor, p_xz: dist.Distribution, q_zx: dist.Distribution):
        x = x.unsqueeze(0)
        log_p_xz = p_xz.log_prob(x)
        log_q_zx = torch.sum(q_zx.log_prob(z), -1)
        log_p_z = torch.sum(self.z_prior.log_prob(z), -1)
        loss = - torch.mean(torch.sum(log_p_xz, -1) + log_p_z - log_q_zx)

        return loss
