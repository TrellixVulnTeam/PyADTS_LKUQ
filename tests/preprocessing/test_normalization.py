import numpy as np

from pyadts.datasets import MSLDataset
from pyadts.preprocessing import min_max_scale, standard_scale, robust_scale, quantile_scale

EPS = 1e-4

dataset = MSLDataset(root='tests/data/msl', train=True, download=True)


def test_min_max_scale():
    dataset_scaled = min_max_scale(dataset, copy=True)
    assert np.abs(np.min(dataset_scaled.to_numpy())) < EPS, f'{np.min(dataset_scaled.to_numpy())}'
    assert np.abs(np.max(dataset_scaled.to_numpy()) - 1) < EPS, f'{np.max(dataset_scaled.to_numpy())}'


def test_standard_scale():
    dataset_scaled = standard_scale(dataset, copy=True)
    assert (np.abs(np.concatenate([np.mean(array, axis=0) for array in dataset_scaled.values])) < EPS).all()
    # assert (np.abs((np.concatenate([np.std(array, axis=0) for array in dataset_scaled.values])) - 1) < EPS).all()


def test_robust_scale():
    dataset_scaled = robust_scale(dataset, copy=True)


def test_quantile_scale():
    dataset_scaled = quantile_scale(dataset, copy=True)
