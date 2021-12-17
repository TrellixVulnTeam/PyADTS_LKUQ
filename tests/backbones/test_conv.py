import torch

from pyadts.backbones import ConvEncoder, ConvDecoder


def test_conv_encoder_decoder():
    for input_size in (500, 512, 511):
        for kernel_sizes in ([3, 3, 3, 3], [4, 4, 5, 5]):
            encoder = ConvEncoder(input_size=input_size, input_channel=2, hidden_channel=16, feature_dim=128,
                                  kernel_sizes=kernel_sizes, strides=[2, 2, 2, 2])
            decoder = ConvDecoder(input_size=input_size, input_channel=2, hidden_channel=16, feature_dim=128,
                                  kernel_sizes=kernel_sizes, strides=[2, 2, 2, 2])

            encoder.cuda()
            decoder.cuda()

            # summary(encoder, (2, input_size), device='cuda')
            # summary(decoder, (128,), device='cuda')

            x = torch.randn(32, 2, input_size).float()
            x = x.cuda()
            print(x.shape, x.device)

            z = encoder(x)
            print(z.shape, z.device)

            x_rec = decoder(z)
            print(x_rec.shape, x_rec.device)

            assert x.shape == x_rec.shape
