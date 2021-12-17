from typing import List

import torch.nn as nn

from pyadts.utils.misc import conv_output_size, conv_transpose_output_size


class ConvEncoder(nn.Module):
    """
    The encoder composed of convolutional networks.

    .. note::
        This model

    Args:
        input_size: Size of timestamps
        input_channel: Number of channels
        hidden_channel: Number of channels that the `head` conv layers outputs
        feature_dim: Dimension of output features
        kernel_sizes: Sizes of conv kernels
        strides: Sizes of strides
    """

    def __init__(self, input_size: int, input_channel: int, hidden_channel: int, feature_dim: int,
                 kernel_sizes: List[int], strides: List[int]):
        super(ConvEncoder, self).__init__()
        assert len(kernel_sizes) == len(strides)
        output_size = input_size

        self.head = nn.Sequential(
            nn.Conv1d(input_channel, hidden_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True)
        )
        output_size = conv_output_size(output_size, 7, 3, 2)

        self.conv_list = []
        prev_channels = hidden_channel
        for i in range(len(kernel_sizes)):
            self.conv_list.append(
                nn.Conv1d(prev_channels, prev_channels * 2, kernel_size=kernel_sizes[i], stride=strides[i],
                          padding=kernel_sizes[i] // 2, bias=False)
            )
            self.conv_list.append(
                nn.BatchNorm1d(prev_channels * 2)
            )
            self.conv_list.append(nn.ReLU(inplace=True))
            prev_channels *= 2
            output_size = conv_output_size(output_size, kernel_sizes[i], kernel_sizes[i] // 2, strides[i])

        self.conv_layers = nn.Sequential(*self.conv_list)

        self.fc = nn.Sequential(
            nn.Linear(output_size * prev_channels, feature_dim)
        )

    def forward(self, x):
        batch_size, *_ = x.shape

        out = self.head(x)
        out = self.conv_layers(out)
        out = out.view(batch_size, -1)
        out = self.fc(out)

        return out


class ConvDecoder(nn.Module):
    """
        The encoder composed of convolutional networks.

        Args:
            input_size: Size of timestamps
            input_channel: Number of channels
            hidden_channel: Number of channels that the `head` conv layers outputs
            feature_dim: Dimension of output features
            kernel_sizes: Sizes of conv kernels
            strides: Sizes of strides
        """

    def __init__(self, input_size: int, input_channel: int, hidden_channel: int, feature_dim: int,
                 kernel_sizes: List[int], strides: List[int]):
        super(ConvDecoder, self).__init__()
        assert len(kernel_sizes) == len(strides)

        output_sizes = [input_size]
        output_sizes.append(conv_output_size(output_sizes[-1], 7, 3, 2))
        for i in range(len(kernel_sizes)):
            output_sizes.append(conv_output_size(output_sizes[-1], kernel_sizes[i], kernel_sizes[i] // 2, strides[i]))
        final_channels = hidden_channel * (2 ** len(kernel_sizes))

        self.final_size, self.final_channels = output_sizes[-1], final_channels

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, self.final_size * self.final_channels)
        )

        output_size = self.final_size
        self.conv_list = []
        prev_channels = final_channels
        for i in reversed(range(len(kernel_sizes))):
            # compute the output padding
            # `output_sizes[i + 1]` denotes the desired output size of the current layer
            # `output_size_wo_padding` denotes the output size without `output_padding` of the current layer
            output_size_wo_padding = conv_transpose_output_size(output_size, kernel_sizes[i], kernel_sizes[i] // 2,
                                                                strides[i])
            output_padding = output_sizes[i + 1] - output_size_wo_padding
            self.conv_list.append(
                nn.ConvTranspose1d(prev_channels, prev_channels // 2, kernel_size=kernel_sizes[i], stride=strides[i],
                                   padding=kernel_sizes[i] // 2, bias=False, output_padding=output_padding)
            )
            self.conv_list.append(
                nn.BatchNorm1d(prev_channels // 2)
            )
            self.conv_list.append(nn.ReLU(inplace=True))
            # update the `output_size` with `output_padding`
            output_size = conv_transpose_output_size(output_size, kernel_sizes[i], kernel_sizes[i] // 2, strides[i],
                                                     output_padding=output_padding)
            prev_channels //= 2

        self.conv_layers = nn.Sequential(*self.conv_list)

        output_padding = output_sizes[0] - conv_transpose_output_size(output_size, 7, 3, 2)
        self.head = nn.Sequential(
            nn.ConvTranspose1d(hidden_channel, input_channel, kernel_size=7, stride=2, padding=3, bias=False,
                               output_padding=output_padding),
        )

    def forward(self, z):
        batch_size, *_ = z.shape

        out = self.fc(z)
        out = out.view(batch_size, self.final_channels, self.final_size)
        out = self.conv_layers(out)
        out = self.head(out)

        return out
