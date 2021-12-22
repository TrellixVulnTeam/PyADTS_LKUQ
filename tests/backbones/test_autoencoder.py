import torch

from pyadts.backbones import (
    Autoencoder, VariationalAutoencoder, ELBO,
    DenseEncoder, DenseDecoder
)


def test_ae():
    model = Autoencoder(encoder=DenseEncoder, decoder=DenseDecoder, input_size=200, feature_dim=128, hidden_size=100,
                        num_layers=3, dropout=0.1, use_batchnorm=True)
    model = model.cuda()

    x = torch.randn(32, 200).to('cuda')
    x_rec, z = model(x)

    print(x.shape, x_rec.shape)

    assert x.shape == x_rec.shape


def test_vae():
    model = VariationalAutoencoder(encoder=DenseEncoder, decoder=DenseDecoder, input_size=200, feature_dim=128,
                                   hidden_size=100,
                                   num_samples=10, num_layers=3, dropout=0.1, use_batchnorm=True)
    model = model.cuda()

    x = torch.randn(32, 200).to('cuda')
    x_dist, z_dist, z = model(x)
    x_rec = x_dist.sample((10,))
    x_rec = x_rec.mean(0)

    print(x.shape, x_rec.shape)

    assert x.shape == x_rec.shape

    criterion = ELBO()
    loss = criterion(x, z, x_dist, z_dist)
    print(loss)


def test_aae():
    pass
