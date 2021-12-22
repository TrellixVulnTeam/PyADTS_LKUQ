import torch
from torchsummary import summary

from pyadts.backbones import GRUEncoder, GRUDecoder, LSTMEncoder, LSTMDecoder


def test_gru():
    encoder = GRUEncoder(input_size=200, hidden_size=100, feature_dim=128, num_layers=3)
    decoder = GRUDecoder(input_size=200, hidden_size=100, feature_dim=128, num_layers=3)
    encoder = encoder.cuda()
    decoder = decoder.cuda()

    summary(encoder, (20, 200), device='cuda')
    summary(decoder, (20, 128), device='cuda')

    x = torch.randn(32, 20, 200).to('cuda')
    z = encoder(x)
    x_rec = decoder(z)

    print(x.shape, z.shape, x_rec.shape)

    assert x.shape == x_rec.shape


def test_lstm():
    encoder = LSTMEncoder(input_size=200, hidden_size=100, feature_dim=128, num_layers=3)
    decoder = LSTMDecoder(input_size=200, hidden_size=100, feature_dim=128, num_layers=3)
    encoder = encoder.cuda()
    decoder = decoder.cuda()

    # summary(encoder, (20, 200), device='cuda')
    # summary(decoder, (20, 128), device='cuda')

    x = torch.randn(32, 20, 200).to('cuda')
    z = encoder(x)
    x_rec = decoder(z)

    print(x.shape, z.shape, x_rec.shape)

    assert x.shape == x_rec.shape
