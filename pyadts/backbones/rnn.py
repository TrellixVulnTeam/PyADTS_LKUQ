import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size: int, feature_dim: int, num_layers: int, dropout: float = 0.3):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size=input_size, hidden_size=feature_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None):
        # x:   (batch, seq_len,    input_size)
        # h_0: (num_layers, batch, hidden_size)

        batch_size, num_epoch, *_ = x.shape

        if h_0 is None:
            h_0 = torch.randn(self.num_layers, batch_size, self.feature_dim)
            h_0 = h_0.cuda(x.device)

        out, h_n = self.gru(x, h_0)

        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        return out, h_n


class LSTM(nn.Module):
    def __init__(self, input_size: int, feature_dim: int, num_layers: int, dropout: float = 0.3):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=feature_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None, c_0: torch.Tensor = None):
        """

        Args:
            x (): of shape (batch, seq_len,    input_size)
            h_0 (): of shape (num_layers, batch, hidden_size)
            c_0 ():

        Returns:

        """

        batch_size, num_epoch, *_ = x.shape

        if h_0 is None:
            h_0 = torch.randn(self.num_layers, batch_size, self.feature_dim)
            h_0 = h_0.to(x.device)

        if c_0 is None:
            c_0 = torch.randn(self.num_layers, batch_size, self.feature_dim)
            c_0 = c_0.to(x.device)

        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        return out, (h_n, c_n)


class GRUEncoder(nn.Module):
    def __init__(self):
        super(GRUEncoder, self).__init__()

    def forward(self, x):
        pass


class GRUDecoder(nn.Module):
    def __init__(self):
        super(GRUDecoder, self).__init__()

    def forward(self, z):
        pass


class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()

    def forward(self, x):
        pass


class LSTMDecoder(nn.Module):
    def __init__(self):
        super(LSTMDecoder, self).__init__()

    def forward(self, z):
        pass
