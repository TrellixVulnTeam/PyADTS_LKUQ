import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from .dataset import KPIDataset


class MLP(nn.Module):
    """

    """

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


class DNN(object):
    """

    """

    def __init__(self, input_size, optimizer='Adam', criterion='BCEWithLogitsLoss', learning_rate=1e-3, dropout=0.5,
                 batch_size=1000, epochs=50):
        self.model = MLP(input_size, dropout).cuda()

        self.__dict__.update(locals())

        summary(self.model, (1, input_size))

    def fit(self, x, y):
        data_set = KPIDataset(x, y, phase='train')
        data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True, num_workers=14)

        optimizer = eval('torch.optim.' + self.optimizer)(self.model.parameters(), lr=self.learning_rate)
        criterion = eval('nn.' + self.criterion)()

        self.model.train()
        for epoch in range(self.epochs):
            total_train_loss = []
            with tqdm(data_loader, desc='EPOCH[%d/%d]' % (epoch + 1, self.epochs)) as loader:
                for x, y in loader:
                    x, y = x.cuda(), y.cuda()

                    optimizer.zero_grad()

                    out = self.model(x)
                    loss = criterion(out, y)

                    loss.backward()
                    optimizer.step()
                    total_train_loss.append(loss.item())

                    loader.set_postfix(
                        {'train_loss': np.nan if len(total_train_loss) == 0 else np.mean(total_train_loss)})

    def predict(self, x):
        data_set = KPIDataset(x, phase='test')
        data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=False, num_workers=14)

        self.model.eval()
        outputs = []
        with torch.no_grad():
            for x in tqdm(data_loader):
                x = x.cuda()
                output = F.sigmoid(self.model(x)).detach().cpu().numpy()
                outputs.append(output)

        return np.concatenate(outputs, axis=0).reshape(-1)
