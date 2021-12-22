from typing import Union

import numpy as np
import torch

from pyadts.generic import Transform

EPS = 1e-6


class MinMaxScaler(Transform):
    def __init__(self):
        super(MinMaxScaler, self).__init__()

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            min_val = np.min(x, axis=-1, keepdims=True)
            max_val = np.max(x, axis=-1, keepdims=True)
            return (x - min_val) / (max_val - min_val + EPS)
        elif isinstance(x, torch.Tensor):
            min_val = torch.min(x, axis=-1, keepdim=True)
            max_val = torch.max(x, axis=-1, keepdim=True)
            return (x - min_val) / (max_val - min_val + EPS)
        else:
            raise ValueError


class StandardScaler(Transform):
    def __init__(self):
        super(StandardScaler, self).__init__()

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            mean_val = np.mean(x, axis=-1, keepdims=True)
            std_val = np.std(x, axis=-1, keepdims=True)
            return (x - mean_val) / (std_val + EPS)
        elif isinstance(x, torch.Tensor):
            mean_val = torch.mean(x, axis=-1, keepdim=True)
            std_val = torch.std(x, axis=-1, keepdim=True)
            return (x - mean_val) / (std_val + EPS)
        else:
            raise ValueError


class SlidingWindow(Transform):
    def __init__(self, window_size: int, stride: int):
        super(SlidingWindow, self).__init__()

        self.window_size = window_size
        self.stride = stride

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        arrays = []
        for i in range(0, x.shape[-1], self.stride):
            if i + self.window_size > x.shape[-1]:
                break
            arrays.append(x[..., i: i + self.window_size])

        if isinstance(x, np.ndarray):
            return np.stack(arrays, axis=-2)
        elif isinstance(x, torch.Tensor):
            return torch.stack(arrays, axis=-2)
        else:
            raise ValueError
