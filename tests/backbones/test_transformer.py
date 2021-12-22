import torch

from pyadts.backbones import TransformerEncoder, TransformerDecoder


def test_transformer():
    encoder = TransformerEncoder(input_size=200, feature_dim=128, num_head=4, hidden_size=2048, dropout=0.1,
                                 activation='relu', num_layers=4, use_pos_enc=True, use_src_mask=True)
    decoder = TransformerDecoder(input_size=200, feature_dim=128, num_head=4, hidden_size=2048, dropout=0.1,
                                 activation='relu', num_layers=4, use_tgt_mask=True)
    encoder = encoder.cuda()
    decoder = decoder.cuda()

    # summary(encoder, (20, 200), device='cuda')
    # summary(decoder, (20, 128), device='cuda')

    x = torch.randn(32, 20, 200).to('cuda')
    z = encoder(x)
    x_rec = decoder(z)

    print(x.shape, z.shape, x_rec.shape)

    assert x.shape == x_rec.shape
