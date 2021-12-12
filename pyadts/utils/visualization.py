from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_series(x: np.ndarray, names: List[str] = None, timestamps: np.ndarray = None,
                label: np.ndarray = None, predictions: np.ndarray = None, title: str = None,
                backend: str = 'matplotlib', style: Union[str, List[str]] = None, fig_size: Tuple[int, int] = None,
                anomaly_color_depth: float = 0.5, ):
    if style is None:
        style = ['ggplot']

    if isinstance(style, str):
        style = [style]

    if fig_size is None:
        fig_size = (12, 7)

    if names is None:
        names = [f'channel-{i}' for i in range(x.shape[-1])]

    if timestamps is None:
        timestamps = np.arange(x.shape[0])

    if backend == 'matplotlib':
        with plt.style.context(style):
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(1, 1, 1)
            for i in range(x.shape[-1]):
                ax.plot(timestamps, x[:, i], color='black', linewidth=0.5, label=names[i])

            # Plot anomalies
            if label is not None:
                for i, xv in enumerate(timestamps[label == 1]):
                    ax.axvline(xv, color='red', lw=1, alpha=anomaly_color_depth, label='Anomaly' if i == 0 else None)

            legend = ax.legend(loc=0, prop={'size': 16})
            legend.get_frame().set_edgecolor('grey')
            legend.get_frame().set_linewidth(2.0)

            if predictions is not None:
                # old_ax = ax
                # ax = fig.add_subplot(2, 1, 2, sharex=old_ax)
                # im = ax.imshow(np.repeat(predictions.reshape(1, -1), repeats=3, axis=0), cmap='bwr', aspect='auto')
                # ax.yaxis.set_visible(False)
                # ax.xaxis.set_visible(False)
                # fig.colorbar(im, ax=old_ax)
                raise NotImplementedError

            if title is not None:
                fig.suptitle(title, fontsize=18)

        return fig
    elif backend == 'plotly':
        raise NotImplementedError
    else:
        raise ValueError

    # with plt.style.context(['science', 'grid']):
    #     grids = GridSpec(ncols=1, nrows=8)
    #     fig = plt.figure(figsize=(12, 5))
    #
    #     axes = []
    #
    #     ax = plt.subplot(grids[:7, :])
    #     ax.plot(np.arange(disp_range), observations[first_anomaly_idx: first_anomaly_idx + disp_range], color='k',
    #             label='Original')
    #     ax.plot(np.arange(disp_range), reconstructions[first_anomaly_idx: first_anomaly_idx + disp_range], 'k--',
    #             label='Reconstruction')
    #     ax.scatter(anomaly_idx, observations[first_anomaly_idx: first_anomaly_idx + disp_range][anomaly_idx], s=5,
    #                c='red', marker='x')
    #     ax.legend(fontsize=16, ncol=2, loc='lower right')
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     axes.append(ax)
    #
    #     ax = plt.subplot(grids[-1, :])
    #     im = ax.imshow(scores[np.newaxis, first_anomaly_idx: first_anomaly_idx + disp_range],
    #                    cmap='bwr', aspect='auto')
    #     ax.yaxis.set_visible(False)
    #     axes.append(ax)
    #
    #     fig.colorbar(im, ax=axes)

    # num_plots = len(x)
    #
    # if num_plots > MAX_PLOT_NUM:
    #     warnings.warn(
    #         'The number of series exceeds the maximum plotting number limit! Only first %d series processed!' % (
    #             MAX_PLOT_NUM))
    #     num_plots = MAX_PLOT_NUM
    #
    # with plt.style.context(['seaborn-whitegrid']):
    #     # fig, axes = plt.subplots(nrows=num_plot, ncols=1, figsize=(12, 4*num_plot), sharex='all')
    #     fig = plt.figure(figsize=(12, 4 * num_plots), tight_layout=True)
    #
    #     ax_prev = None
    #     for i in range(num_plots):
    #         # axes[i].plot(data_df.index, data_df['value'], color='black', linewidth=0.5, label='series')
    #         ax = fig.add_subplot(num_plots, 1, i + 1, sharex=ax_prev)
    #         ax_prev = ax
    #         ax.visualize(timestamps[i], x[i], color='black', linewidth=0.5, label=names[i])
    #
    #         # Plot anomalies
    #         if label is not None:
    #             for xv in timestamps[label == 1]:
    #                 ax.axvline(xv, color='orange', lw=1, alpha=anomaly_color_depth)
    #
    #         if predictions is not None:
    #             ax.visualize(timestamps[predictions[i] == 1], x[predictions == 1], linewidth=0, color='red', marker='x',
    #                          markersize=5,
    #                          label='predictions')
    #
    #         # legend = axes[i].legend(loc=0, prop={'size': 16})
    #         legend = ax.legend(loc=0, prop={'size': 16})
    #         legend.get_frame().set_edgecolor('grey')
    #         legend.get_frame().set_linewidth(2.0)
    #
    #     if title is not None:
    #         fig.suptitle(title, fontsize=18)
    #
    # return fig


def plot_roc(scores: np.ndarray, labels: np.ndarray, backend: str = 'matplotlib', style: str = 'science'):
    assert scores.shape == labels.shape
    assert scores.ndim == 1

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

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


def plot_space(x: List[np.ndarray], label: np.ndarray, backend: str = 'matplotlib'):
    fig = plt.figure(figsize=(12, 4.5))
    # ax = fig.add_subplot(121, projection='3d')
    # sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
    #                 c=np.log(avg_codisp.sort_index().values),
    #                 cmap='gnuplot2')
    # plt.title('log(CoDisp)')
    # ax = fig.add_subplot(122, projection='3d')
    # sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
    #                 linewidths=0.1, edgecolors='k',
    #                 c=(avg_codisp >= threshold).astype(float),
    #                 cmap='cool')
    # plt.title('CoDisp above 99.5th percentile')

    return fig
