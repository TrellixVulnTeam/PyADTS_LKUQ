import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

MAX_PLOT_NUM = 4


def plot_series(x: np.ndarray, names: List[str] = None, timestamps: np.ndarray = None, label: np.ndarray = None,
                predictions: np.ndarray = None, title: str = None, anomaly_color_depth: float = 0.2):
    # check `x`
    if x.ndim == 1:
        x = x.reshape(1, -1)
    else:
        assert x.ndim == 2

    # check `names`
    if names is not None:
        assert len(names) == x.shape[-2]
    else:
        names = [f'series{i}' for i in range(x.shape[-2])]

    # check `timestamps`
    if timestamps is not None:
        try:
            timestamps = timestamps.reshape(x.shape)
        except ValueError:
            raise ValueError('The shape of `timestamp` is not consistent  with `x`!')
    else:
        timestamps = np.repeat(np.arange(x.shape[-1]).reshape(1, -1), repeats=(x.shape[-2], 1))

    # check `label`
    if label is not None:
        try:
            label = label.reshape(x.shape)
        except ValueError:
            raise ValueError('The shape of `label` is not consistent  with `x`!')

    # check `predictions`
    if predictions is not None:
        try:
            predictions = predictions.reshape(x.shape)
        except ValueError:
            raise ValueError('The shape of `predictions` is not consistent  with `x`!')

    num_plot = x.shape[-2]  ## x: (channel, timestamps)

    if num_plot > MAX_PLOT_NUM:
        warnings.warn(
            'The number of series exceeds the maximum plotting number limit! Only first %d series processed!' % (
                MAX_PLOT_NUM))
        num_plot = MAX_PLOT_NUM

    with plt.style.context(['seaborn-whitegrid']):
        # fig, axes = plt.subplots(nrows=num_plot, ncols=1, figsize=(12, 4*num_plot), sharex='all')
        fig = plt.figure(figsize=(12, 4 * num_plot), tight_layout=True)

        ax_prev = None
        for i in range(num_plot):
            # axes[i].plot(data_df.index, data_df['value'], color='black', linewidth=0.5, label='series')
            ax = fig.add_subplot(num_plot, 1, i + 1, sharex=ax_prev)
            ax_prev = ax
            ax.plot(timestamps[i], x[i], color='black', linewidth=0.5, label=names[i])

            # Plot anomalies
            if label is not None:
                for xv in timestamps[label == 1]:
                    ax.axvline(xv, color='orange', lw=1, alpha=anomaly_color_depth)

            if predictions is not None:
                ax.plot(timestamps[predictions[i] == 1], x[predictions == 1], linewidth=0, color='red', marker='x',
                        markersize=5,
                        label='predictions')

            # legend = axes[i].legend(loc=0, prop={'size': 16})
            legend = ax.legend(loc=0, prop={'size': 16})
            legend.get_frame().set_edgecolor('grey')
            legend.get_frame().set_linewidth(2.0)

        if title is not None:
            fig.suptitle(title, fontsize=18)
    return fig


def plot_roc(scores: np.ndarray, labels: np.ndarray):
    assert scores.shape == labels.shape
    assert scores.ndim == 1

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
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
    plt.show()
