import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

MAX_PLOT_NUM = 4


def plot_series(data_df: pd.DataFrame, meta_df: pd.DataFrame = None, predictions: pd.DataFrame = None,
                title: str = None, anomaly_color_depth: float = 0.2):
    num_plot = data_df.shape[1]
    if num_plot > MAX_PLOT_NUM:
        warnings.warn(
            'The number of series exceeds the maximum plotting number limit! Only first %d series processed!' % (
                MAX_PLOT_NUM))
        num_plot = MAX_PLOT_NUM

    if predictions is not None:
        assert predictions.shape == data_df.shape

    with plt.style.context(['seaborn-whitegrid']):
        # fig, axes = plt.subplots(nrows=num_plot, ncols=1, figsize=(12, 4*num_plot), sharex='all')
        fig = plt.figure(figsize=(12, 4 * num_plot), tight_layout=True)

        ax_prev = None
        for i in range(num_plot):
            # axes[i].plot(data_df.index, data_df['value'], color='black', linewidth=0.5, label='series')
            ax = fig.add_subplot(num_plot, 1, i + 1, sharex=ax_prev)
            ax_prev = ax
            ax.plot(data_df.index, data_df.iloc[:, i], color='black', linewidth=0.5, label=data_df.columns[i])

            # Plot anomalies
            if meta_df is not None:
                date_index = data_df.index
                label = meta_df['label']
                value = data_df.iloc[:, i]
                for xv in date_index[label == 1]:
                    # axes[i].axvline(xv, color='orange', lw=1, alpha=0.1)
                    ax.axvline(xv, color='orange', lw=1, alpha=anomaly_color_depth)
                # axes[i].plot(date_index[label == 1], value[label == 1], linewidth=0, color='red', marker='x', markersize=5, label='anomalies')

            if predictions is not None:
                value = data_df.iloc[:, i]
                ax.plot(date_index[predictions.iloc[:, i] == 1], value[(predictions.iloc[:, i] == 1).values], linewidth=0, color='red', marker='x', markersize=5,
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
