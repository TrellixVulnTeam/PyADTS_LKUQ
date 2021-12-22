import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    adapt from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        input_size ():
        dropout ():
        max_len (): determines how far the position can have an effect on a token (window)
    """

    def __init__(self, input_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2) * (-np.log(10000.0) / input_size))
        pe = torch.zeros(max_len, 1, input_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        Args:
            x (): shape [batch_size, seq_len, embedding_dim]

        Returns:

        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size: int, feature_dim: int, hidden_size: int = 2048, num_head: int = 4,
                 dropout: float = 0.1, activation: str = 'relu', num_layers: int = 4, norm: nn.Module = None,
                 use_pos_enc: bool = True, use_src_mask: bool = True, max_len: int = 5000):
        """
        A transformer encoder with positional encoding and source masking.

        Args:
            feature_dim ():
            num_head ():
            hidden_size ():
            dropout ():
            activation ():
            num_layers ():
            norm ():
            use_pos_enc ():
            use_src_mask ():
        """
        super(TransformerEncoder, self).__init__()

        self.input_size = input_size
        self.feature_dim = feature_dim
        self.use_pos_enc = use_pos_enc
        self.use_src_mask = use_src_mask

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=num_head,
                dim_feedforward=hidden_size,
                dropout=dropout,
                activation=activation,
                batch_first=True
            ),
            num_layers=num_layers, norm=norm)
        self.fc = nn.Sequential(
            nn.Linear(input_size, feature_dim)
        )

        if use_pos_enc:
            self.pos_encoder = PositionalEncoding(input_size=input_size, max_len=max_len, dropout=dropout)

    def forward(self, src):
        # src (batch_size, seq_len, input_size)

        if self.use_pos_enc:
            src = src * np.sqrt(self.input_size)
            src = self.pos_encoder(src)

        if self.use_src_mask:
            # EX for size=5:
            # [[0., -inf, -inf, -inf, -inf],
            #  [0.,   0., -inf, -inf, -inf],
            #  [0.,   0.,   0., -inf, -inf],
            #  [0.,   0.,   0.,   0., -inf],
            #  [0.,   0.,   0.,   0.,   0.]]
            mask = torch.triu(torch.ones(src.size(1), src.size(1)) * float('-inf'), diagonal=1)
            mask = mask.to(src.device)
        else:
            mask = None

        out = self.encoder(src, mask=mask)
        out = self.fc(out)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, input_size: int, feature_dim: int, hidden_size: int = 2048, num_head: int = 4,
                 dropout: float = 0.1, activation: str = 'relu', num_layers: int = 4, norm: nn.Module = None,
                 use_tgt_mask: bool = True):
        super(TransformerDecoder, self).__init__()

        self.feature_dim = feature_dim
        self.use_tgt_mask = use_tgt_mask

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_head,
                dim_feedforward=hidden_size,
                dropout=dropout,
                activation=activation,
                batch_first=True
            ),
            num_layers=num_layers, norm=norm)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, input_size)
        )

    def forward(self, z):
        if self.use_tgt_mask:
            # EX for size=5:
            # [[0., -inf, -inf, -inf, -inf],
            #  [0.,   0., -inf, -inf, -inf],
            #  [0.,   0.,   0., -inf, -inf],
            #  [0.,   0.,   0.,   0., -inf],
            #  [0.,   0.,   0.,   0.,   0.]]
            mask = torch.triu(torch.ones(z.size(1), z.size(1)) * float('-inf'), diagonal=1)
            mask = mask.to(z.device)
        else:
            mask = None

        out = self.decoder(z, mask=mask)
        out = self.fc(out)

        return out
