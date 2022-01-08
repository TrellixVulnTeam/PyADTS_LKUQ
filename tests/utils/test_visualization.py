import numpy as np
from scipy.signal import savgol_filter

from pyadts.utils.visualization import plot_series, plot_space


def test_plot_series():
    x = np.arange(1000)
    y = np.stack([np.sin(x / 3) * 2 + np.cos(x), np.sin(x) + np.cos(x / 2) * 2 - 1, np.sin(x) + np.cos(x / 2)], axis=-1)
    ano_idx = np.random.choice(np.arange(x.shape[0]), size=50, replace=False)
    y[ano_idx, :] += (np.random.randn(50).reshape(-1, 1))
    label = np.zeros(x.shape[0])
    label[ano_idx] = 1
    predictions = np.zeros(x.shape[0])
    predictions[ano_idx] = np.abs(np.random.randn(50))
    predictions = savgol_filter(predictions, window_length=21, polyorder=1)

    fig = plot_series(y, timestamps=x, labels=label, predictions=predictions)
    fig.show()


def test_plot_space():
    x = np.arange(1000)
    y = np.stack([np.sin(x / 3) * 2 + np.cos(x), np.sin(x) + np.cos(x / 2) * 2 - 1, np.sin(x) + np.cos(x / 2)], axis=-1)
    ano_idx = np.random.choice(np.arange(x.shape[0]), size=50, replace=False)
    y[ano_idx, :] += (np.random.randn(50).reshape(-1, 1))
    label = np.zeros(x.shape[0])
    label[ano_idx] = 1

    fig = plot_space(y, label, decomposition_dim=3, decomposition_method='pca')
    fig.show()
