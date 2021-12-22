"""
@Time    : 2021/10/25 15:13
@File    : donut.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from pyadts.generic import Detector, TimeSeriesDataset


def mcmc_missing_imputation(observe_normal, vae: nn.Module, n_iteration=10, **inputs):
    assert "observe_x" in inputs
    observe_x = inputs["observe_x"]

    if isinstance(observe_x, torch.Tensor):
        test_x = torch.Tensor(observe_x.data)
    else:
        test_x = torch.Tensor(observe_x)
    with torch.no_grad():
        for mcmc_step in range(n_iteration):
            p_xz, _, _ = vae(**inputs, n_sample=1)
            test_x[observe_normal == 0.] = p_xz.sample()[0][observe_normal == 0.]
    return test_x


def m_elbo(observe_x, observe_z, normal, p_xz: dist.Distribution, q_zx: dist.Distribution, p_z: dist.Distribution):
    """
    :param observe_x: (batch_size, x_dims)
    :param observe_z: (sample_size, batch_size, z_dims) or (batch_size, z_dims,)
    :param normal: (batch_size, x_dims)
    :param p_xz: samples in shape (sample_size, batch_size, x_dims)
    :param q_zx: samples in shape (sample_size, batch_size, z_dims)
    :param p_z: samples in shape (z_dims, )
    :return:
    """
    observe_x = torch.unsqueeze(observe_x, 0)  # (1, batch_size, x_dims)
    normal = torch.unsqueeze(normal, 0)  # (1, batch_size, x_dims)
    log_p_xz = p_xz.log_prob(observe_x)  # (1, batch_size, x_dims)
    if observe_z.dim() == 2:
        torch.unsqueeze(observe_z, 0, observe_z)  # (sample_size, batch_size, z_dims)
    # noinspection PyArgumentList
    log_q_zx = torch.sum(q_zx.log_prob(observe_z), -1)  # (sample_size, batch_size)
    # noinspection PyArgumentList
    log_p_z = torch.sum(p_z.log_prob(observe_z), -1)  # (sample_size, batch_size)
    # noinspection PyArgumentList
    radio = (torch.sum(normal, -1) / float(normal.size()[-1]))  # (1, batch_size)
    # noinspection PyArgumentList
    return - torch.mean(torch.sum(log_p_xz * normal, -1) + log_p_z * radio - log_q_zx)


class Donut(Detector):
    def __init__(self):
        super(Donut, self).__init__()

    def fit(self, x: Union[np.ndarray, TimeSeriesDataset], y=None):
        pass

    def predict(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass

    def score(self, x: Union[np.ndarray, TimeSeriesDataset]):
        pass
