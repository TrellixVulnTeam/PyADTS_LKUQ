"""
@Time    : 2021/10/26 0:14
@File    : aae.py
@Software: PyCharm
@Desc    :
"""
from typing import Union

import numpy as np

from pyadts.generic import Detector, TimeSeriesDataset


from pyadts.backbones import (
    AdversarialAutoencoder,
    DenseEncoder, DenseDecoder, ConvEncoder, ConvDecoder,
    GRUEncoder, GRUDecoder, LSTMEncoder, LSTMDecoder, TransformerEncoder, TransformerDecoder
)
from pyadts.generic import Detector, TimeSeriesDataset
import numpy as np
import torch
from typing import Union
from pyadts.utils.data import any_to_tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm.std import tqdm
import torch.nn as nn
class AAE(Detector):
    def __init__(self, arch: str, input_size: int, feature_dim: int, hidden_size: int, num_samples: int = 1,
                 batch_size: int = 32, data_lr: float = 1e-3,dis_lr: float = 1e-3,gen_lr: float = 1e-3, epochs: int = 10,optim: str = 'adam', optim_args=None,
                 device: str = None, verbose: bool = True, **kwargs):
        super(AAE, self).__init__()

        if optim_args is None:
            optim_args = {}
        self.batch_size = batch_size
        self.data_lr = data_lr
        self.dis_lr=dis_lr
        self.gen_lr=gen_lr
        self.epochs = epochs
        self.verbose = verbose
        self.ae_criterion = nn.MSELoss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.register_buffer("label_real", torch.ones(batch_size, 1))
        self.register_buffer("label_fake", torch.zeros(batch_size, 1))
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # setup base architecture
        if arch == 'dense':
            encoder = DenseEncoder
            decoder = DenseDecoder
            discriminator=DenseDecoder
        elif arch == 'conv':
            encoder = ConvEncoder
            decoder = ConvDecoder
            discriminator = DenseDecoder
        elif arch == 'gru':
            encoder = GRUEncoder
            decoder = GRUDecoder
            discriminator = GRUEncoder
        elif arch == 'lstm':
            encoder = LSTMEncoder
            decoder = LSTMDecoder
            discriminator = LSTMDecoder
        elif arch == 'transformer':
            encoder = TransformerEncoder
            decoder = TransformerDecoder
            discriminator = TransformerDecoder
        else:
            raise ValueError

        # setup vae model
        self.model = AdversarialAutoencoder(encoder, decoder, discriminator,input_size, feature_dim, hidden_size,**kwargs)
        self.model = self.model.to(self.device)

        # setup optimizer
        if optim == 'sgd':
            self.autoencoder_optimizer = torch.optim.SGD(self.model.parameters(), lr=data_lr, **optim_args)
            self.dis_optimizer = torch.optim.SGD(self.model.parameters(), lr=dis_lr, **optim_args)
            self.gen_optimizer = torch.optim.SGD(self.model.parameters(), lr=gen_lr, **optim_args)
        elif optim == 'adam':
            self.data_optimizer = torch.optim.Adam(self.model.parameters(), lr=data_lr, **optim_args)
            self.dis_optimizer = torch.optim.Adam(self.model.parameters(), lr=dis_lr, **optim_args)
            self.gen_optimizer = torch.optim.Adam(self.model.parameters(), lr=gen_lr, **optim_args)
        elif optim == 'adamw':
            self.data_optimizer = torch.optim.AdamW(self.model.parameters(), lr=data_lr, **optim_args)
            self.dis_optimizer = torch.optim.AdamW(self.model.parameters(), lr=dis_lr, **optim_args)
            self.gen_optimizer = torch.optim.AdamW(self.model.parameters(), lr=gen_lr, **optim_args)
        else:
            raise ValueError

    def fit(self,x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], y=None):
        x = any_to_tensor(x, self.device)
        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, drop_last=True, pin_memory=True, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            if self.verbose:
                data_iter = tqdm(data_loader, desc=f'EPOCH [{epoch + 1}/{self.epochs}]')
                rec_losses = []
                dis_losses=[]
                gen_losses=[]
            else:
                data_iter = data_loader

            for x in data_iter:
                x = x.to(self.device)
            x_rec, z = self.model(x)

            ##################################################################################
            # Data Model loss
            ##################################################################################
            rec_loss = self.ae_criterion(x_rec, x)
            self.autoencoder_optimizer.zero_grad()
            rec_loss.backward()
            self.autoencoder_optimizer.step()

            ##################################################################################
            # Data discrimination
            ##################################################################################
            dis_loss = self.adversarial_criterion(self.discriminator(x_rec), self.label_fake) + \
                            self.adversarial_criterion(self.discriminator(x), self.label_real)
            self.dis_optimizer.zero_grad()
            dis_loss.backward()
            self.dis_optimizer.step()

            ##################################################################################
            # Generator
            ##################################################################################
            gen_loss = self.adversarial_criterion(self.data_discriminator(x_rec), self.label_real)
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

            if self.verbose:
                rec_losses.append(rec_loss.item())
                dis_losses.append(dis_loss.item())
                gen_losses.append(gen_loss.item())
                data_iter.set_postfix({'loss': np.mean(rec_loss)})
                data_iter.set_postfix({'loss': np.mean(dis_loss)})
                data_iter.set_postfix({'loss': np.mean(gen_loss)})

    def score(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor]):
        x = any_to_tensor(x, self.device)
        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, drop_last=False, pin_memory=True, shuffle=False)
        self.model.eval()
        if self.verbose:
            data_iter = tqdm(data_loader)
        else:
            data_iter = data_loader
        scores = []
        with torch.no_grad():
            for x in data_iter:
                x = x.to(self.device)
                x_rec, z = self.model(x)
                rec_loss = nn.MSELoss(x_rec, x)
                scores.append(np.mean(rec_loss.item(), axis=0))
            scores = np.concatenate(scores)
        return scores
