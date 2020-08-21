import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base import BaseModel
from ...data.utils import to_tensor_dataset


class MLP(nn.Module):
    def __init__(self, input_size, dropout=0.5):
        super(MLP, self).__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_size // 4, 1),
        )

    def forward(self, x):
        out = self.layers(x)

        return out


class DNN(BaseModel):
    def __init__(self, input_size, device='cuda', optimizer='Adam', criterion='BCEWithLogitsLoss', learning_rate=1e-3,
                 dropout=0.5, batch_size=1000, epochs=50):
        super().__init__()

        if device.startswith('cuda'):
            assert torch.cuda.is_available()
        elif device == 'cpu':
            pass
        else:
            raise ValueError('Invalid device setting!')
        self.device = device

        self.model = MLP(input_size, dropout).to(device)
        self.batch_size = batch_size
        self.optimizer = eval(optimizer)(self.model.parameters(), lr=learning_rate)
        self.criterion = eval(criterion)()
        self.dropout = dropout
        self.epochs = epochs

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        self.store_train_data(x, y)

        data_set = to_tensor_dataset(x, y)
        data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

        self.model.train()
        for epoch in range(self.epochs):
            total_train_loss = []
            with tqdm(data_loader, desc='EPOCH[%d/%d]' % (epoch + 1, self.epochs)) as loader:
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer.zero_grad()

                    out = self.model(x)
                    loss = self.criterion(out, y)

                    loss.backward()
                    self.optimizer.step()
                    total_train_loss.append(loss.item())

                    loader.set_postfix(
                        {'train_loss': np.nan if len(total_train_loss) == 0 else np.mean(total_train_loss)})

    def predict_score(self, x: np.ndarray):
        self.check_fitted()

        data_set = to_tensor_dataset(x)
        data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.model.eval()
        outputs = []
        with torch.no_grad():
            for x in tqdm(data_loader):
                x = x.cuda()

                output = self.model(x)
                output = torch.sigmoid(output)
                output = output.detach().cpu().numpy()
                outputs.append(output)

        return np.concatenate(outputs, axis=0).reshape(-1)
