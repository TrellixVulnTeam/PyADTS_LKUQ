import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, feature_dim: int, num_layers: int, dropout: float = 0.3):
        super(GRUEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, feature_dim)
        )

        self.__init_weights()

    def __init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x: torch.Tensor):
        # x:   (batch, seq_len,    input_size)
        # h_0: (num_layers, batch, hidden_size)

        batch_size, num_epoch, *_ = x.shape

        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
        h_0 = h_0.to(x.device)

        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, h_n = self.gru(x, h_0)

        out = self.fc(out)

        return out


class GRUDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, feature_dim: int, num_layers: int, dropout: float = 0.3):
        super(GRUDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, input_size)
        )

        self.__init_weights()

    def __init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, z: torch.Tensor):
        # z:   (batch, seq_len,    feature_dim)
        batch_size, num_epoch, *_ = z.shape

        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
        h_0 = h_0.to(z.device)

        out, h_n = self.gru(z, h_0)
        out = self.fc(out)

        return out


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, feature_dim: int, num_layers: int, dropout: float = 0.3):
        super(LSTMEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, feature_dim)
        )

        self.__init_weights()

    def __init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x: torch.Tensor):
        batch_size, num_epoch, *_ = x.shape

        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
        h_0 = h_0.to(x.device)

        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
        c_0 = c_0.to(x.device)

        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.fc(out)

        return out


class LSTMDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, feature_dim: int, num_layers: int, dropout: float = 0.3):
        super(LSTMDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, input_size)
        )

        self.__init_weights()

    def __init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, z: torch.Tensor):
        batch_size, num_epoch, *_ = z.shape

        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
        h_0 = h_0.to(z.device)

        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
        c_0 = c_0.to(z.device)

        out, (h_n, c_n) = self.lstm(z, (h_0, c_0))
        out = self.fc(out)

        return out
