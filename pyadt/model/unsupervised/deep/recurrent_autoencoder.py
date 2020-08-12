import torch
import torch.nn as nn

from ...base import BaseModel


class RecurrentAutoencoder(BaseModel):
    def __init__(self):
        super(RecurrentAutoencoder, self).__init__()

