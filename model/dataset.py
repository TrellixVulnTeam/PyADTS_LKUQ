import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset


class KPIDataset(Dataset):
    """
    Dataset for KPI anomaly detection
    """
    def __init__(self, x, *y, phase='train'):
        super(KPIDataset, self).__init__()
        self.dataset = x
        if phase == 'test':
            assert(len(y)==0)
        else:
            assert(len(y)==1)
            self.label = y[0]
        self.phase = phase
    
    def __getitem__(self, idx):
        x = self.dataset[idx,:].astype(np.float32)
        if self.phase == 'train':
            y = self.label[idx:idx+1].astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(y)
        else:
            return torch.from_numpy(x)

    def __len__(self):
        return self.dataset.shape[0]
