import warnings

import matplotlib.pyplot as plt
import pandas as pd
from pandas import plotting


MAX_PLOT_NUM = 4


def plot_series(data_df: pd.DataFrame, meta_df: pd.DataFrame=None, title: str = None, plot_vline:bool=True):
    num_plot = data_df.shape[1]
    if num_plot > MAX_PLOT_NUM:
        warnings.warn('The number of series exceeds the maximum plotting number limit! Only first %d series processed!'%(MAX_PLOT_NUM))
        num_plot = MAX_PLOT_NUM

    with plt.style.context(['seaborn-whitegrid']):
        # fig, axes = plt.subplots(nrows=num_plot, ncols=1, figsize=(12, 4*num_plot), sharex='all')
        fig = plt.figure(figsize=(12, 4*num_plot), tight_layout=True)

        ax_prev = None
        for i in range(num_plot):
            # axes[i].plot(data_df.index, data_df['value'], color='black', linewidth=0.5, label='series')
            ax = fig.add_subplot(num_plot, 1, i+1, sharex=ax_prev)
            ax_prev = ax
            ax.plot(data_df.index, data_df.iloc[:, i], color='black', linewidth=0.5, label=data_df.columns[i])

            # Plot anomalies
            if meta_df is not None:
                date_index = data_df.index
                label = meta_df['label']
                value = data_df.iloc[:, i]
                if plot_vline:
                    for xv in date_index[label == 1]:
                        # axes[i].axvline(xv, color='orange', lw=1, alpha=0.1)
                        ax.axvline(xv, color='orange', lw=1, alpha=0.1)
                # axes[i].plot(date_index[label == 1], value[label == 1], linewidth=0, color='red', marker='x', markersize=5, label='anomalies')
                ax.plot(date_index[label == 1], value[label == 1], linewidth=0, color='red', marker='x', markersize=5, label='anomalies')

            # legend = axes[i].legend(loc=0, prop={'size': 16})
            legend = ax.legend(loc=0, prop={'size': 16})
            legend.get_frame().set_edgecolor('grey')
            legend.get_frame().set_linewidth(2.0)

        if title is not None:
            fig.suptitle(title, fontsize=18)
    return fig


def plot_lag(data_df: pd.DataFrame, lag:int = 1, title: str=None):
    with plt.style.context(['seaborn-whitegrid']):
        ax = plotting.lag_plot(data_df['value'], lag=lag)

        if title is not None:
            ax.set_title(title)
