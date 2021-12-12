import numpy as np

from pyadts.utils.visualization import plot_series


def test_plot_series():
    x = np.arange(1000)
    y = np.stack([np.sin(x) * 2 + np.cos(x), np.sin(x) + np.cos(x) * 2 - 1], axis=-1)
    ano_idx = np.random.choice(np.arange(x.shape[0]), size=50, replace=False)
    y[ano_idx, :] += (np.random.randn(50).reshape(-1, 1) * 2)
    label = np.zeros(x.shape[0])
    label[ano_idx] = 1
    predictions = np.zeros(x.shape[0])
    predictions[ano_idx] = np.abs(np.random.randn(50))

    fig = plot_series(y, timestamps=x, label=label)
    fig.show()
