"""
@Time    : 2021/10/26 0:14
@File    : vae.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.std import tqdm

from pyadts.backbones import (
    VariationalAutoencoder, ELBO,
    DenseEncoder, DenseDecoder, ConvEncoder, ConvDecoder,
    GRUEncoder, GRUDecoder, LSTMEncoder, LSTMDecoder, TransformerEncoder, TransformerDecoder
)
from pyadts.generic import Detector, TimeSeriesDataset
from pyadts.utils.data import any_to_tensor


class VAE(Detector):
    def __init__(self, arch: str, input_size: int, feature_dim: int, hidden_size: int, num_samples: int = 1,
                 batch_size: int = 32, lr: float = 1e-3, epochs: int = 10, optim: str = 'adam', optim_args=None,
                 device: str = None, verbose: bool = True, **kwargs):
        super(VAE, self).__init__()

        if optim_args is None:
            optim_args = {}
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # setup base architecture
        if arch == 'dense':
            encoder = DenseEncoder
            decoder = DenseDecoder
        elif arch == 'conv':
            encoder = ConvEncoder
            decoder = ConvDecoder
        elif arch == 'gru':
            encoder = GRUEncoder
            decoder = GRUDecoder
        elif arch == 'lstm':
            encoder = LSTMEncoder
            decoder = LSTMDecoder
        elif arch == 'transformer':
            encoder = TransformerEncoder
            decoder = TransformerDecoder
        else:
            raise ValueError

        # setup vae model
        self.model = VariationalAutoencoder(encoder, decoder, input_size, feature_dim, hidden_size, num_samples,
                                            **kwargs)
        self.model = self.model.to(self.device)

        # setup optimizer
        if optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, **optim_args)
        elif optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, **optim_args)
        elif optim == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, **optim_args)
        else:
            raise ValueError

        # setup loss function
        self.criterion = ELBO()

    def fit(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], y=None):
        x = any_to_tensor(x, self.device)
        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, drop_last=True, pin_memory=True, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            if self.verbose:
                data_iter = tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{self.epochs}]')
                losses = []
            else:
                data_iter = data_loader

            for x in data_iter:
                x = x.to(self.device)

            p_xz, q_zx, z = self.model(x)
            loss = self.criterion(x, z, p_xz, q_zx)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.verbose:
                losses.append(loss.item())
                data_iter.set_postfix({'loss': np.mean(losses)})

    def score(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor]):
        x = any_to_tensor(x, self.device)
        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, drop_last=False, pin_memory=True, shuffle=False)

        scores = []

        self.model.eval()
        if self.verbose:
            data_iter = tqdm(data_loader)
        else:
            data_iter = data_loader

        with torch.no_grad():
            for x in data_iter:
                x = x.to(self.device)

            p_xz, q_zx, z = self.model(x)
            loss = self.criterion(x, z, p_xz, q_zx)

            log_p_xz = p_xz.log_prob(x).cpu().numpy()

            scores.append(np.mean(log_p_xz, axis=0))

        scores = np.concatenate(scores)
        return scores

