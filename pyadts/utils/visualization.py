import warnings
from datetime import datetime
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

MAX_PLOT_NUM = 5


def plot_series(x: np.ndarray, names: List[str] = None, timestamps: np.ndarray = None,
                labels: np.ndarray = None, predictions: np.ndarray = None, title: str = None,
                backend: str = 'matplotlib', style: Union[str, List[str]] = None, fig_size: Tuple[int, int] = None):
    num_plots = x.shape[-1]
    if num_plots > MAX_PLOT_NUM:
        warnings.warn('The number of series exceeds the recommended maximum plotting number! '
                      'The layout of the figure will be unpredictable!')

    if style is None:
        style = ['seaborn-whitegrid']

    if isinstance(style, str):
        style = [style]

    if fig_size is None:
        fig_size = (15, 8)

    if names is None:
        names = [f'channel-{i}' for i in range(x.shape[-1])]

    if timestamps is None:
        timestamps = np.arange(x.shape[0])

    datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]
    for i, dt in enumerate(datetimes):
        if dt > datetime.now() or dt < datetime(1800, 1, 1, 0, 0, 0, 0):
            warnings.warn(f'The timestamp {timestamps[i]} is not valid.')
            datetimes = timestamps
            break

    if backend == 'matplotlib':
        with plt.style.context(style):

            grids = GridSpec(ncols=1, nrows=num_plots * 2 + 1)
            fig = plt.figure(figsize=fig_size)
            axes = []

            for i in range(x.shape[-1]):
                ax = plt.subplot(grids[i * 2: (i + 1) * 2, :])
                ax.plot(datetimes, x[:, i], color='black', linewidth=1.5, label=names[i])
                if labels is not None:
                    for j, xv in enumerate(datetimes[labels == 1]):
                        ax.axvline(xv, color='red', lw=1, alpha=0.5)
                ax.set_ylim(np.min(x[:, i]) * 1.1, np.max(x[:, i]) * 1.1)
                if i != x.shape[-1] - 1:
                    ax.xaxis.set_visible(False)
                legend = ax.legend(loc=3, prop={'size': 14})
                legend.get_frame().set_edgecolor('grey')
                legend.get_frame().set_linewidth(2.0)
                axes.append(ax)

            if predictions is not None:
                ax = plt.subplot(grids[-1, :])
                ax = plt.subplot(grids[-1, :])
                im = ax.imshow(predictions[np.newaxis, :], cmap='bwr', aspect='auto')
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                axes.append(ax)

                fig.colorbar(im, ax=axes)

            if title is not None:
                fig.suptitle(title, fontsize=18)

            # fig.tight_layout()

        return fig
    elif backend == 'plotly':
        raise NotImplementedError
    else:
        raise ValueError


def plot_roc(scores: np.ndarray, labels: np.ndarray, backend: str = 'matplotlib', style: str = 'science'):
    assert scores.shape == labels.shape
    assert scores.ndim == 1

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    if backend == 'matploblib':
        fig = plt.figure(figsize=(12, 4.5))

        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        return fig
    elif backend == 'plotly':
        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        return fig
    else:
        raise ValueError


def plot_space(x: np.ndarray, label: np.ndarray, decomposition_method: str = 'pca', decomposition_dim: int = 3,
               backend: str = 'matplotlib'):
    if x.shape[-1] > decomposition_dim:
        if decomposition_method == 'pca':
            projector = PCA(n_components=decomposition_dim)
        elif decomposition_method == 'tsne':
            projector = TSNE(n_components=decomposition_dim)
        else:
            raise ValueError

        x_decomp = projector.fit_transform(x)
    else:
        x_decomp = x

    if backend == 'matplotlib':
        fig = plt.figure(figsize=(8, 6))
        if decomposition_dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_decomp[label == 0, 0], x_decomp[label == 0, 1], x_decomp[label == 0, 2],
                       linewidth=0.1, edgecolor='k', c='tab:blue', label='Normal')
            ax.scatter(x_decomp[label == 1, 0], x_decomp[label == 1, 1], x_decomp[label == 1, 2],
                       linewidth=0.1, edgecolor='k', c='tab:red', label='Anomaly')
            ax.legend()
        elif decomposition_dim == 2:
            ax = fig.add_subplot(111)
            ax.scatter(x_decomp[label == 0, 0], x_decomp[label == 0, 1], linewidth=0.1, edgecolor='k',
                       c='tab:blue', label='Normal')
            ax.scatter(x_decomp[label == 1, 0], x_decomp[label == 1, 1], linewidth=0.1, edgecolor='k',
                       c='tab:red', label='Anomaly')
            ax.legend()
        else:
            raise ValueError
    elif backend == 'plotly':
        if decomposition_dim == 3:
            fig = px.scatter_3d(x_decomp[label == 0, 0], x_decomp[label == 0, 1], x_decomp[label == 0, 2], opacity=0.7)
        elif decomposition_dim == 2:
            fig = px.scatter(x_decomp[label == 0, 0], x_decomp[label == 0, 1], opacity=0.7)
        else:
            raise ValueError
    else:
        raise ValueError

    return fig
