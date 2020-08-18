from typing import Union

import numpy as np
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
import seaborn as sns


def plot_series(data_df: pd.DataFrame, meta_df: pd.DataFrame=None, title: str = None, plot_vline:bool=True):
    with plt.style.context(['seaborn-whitegrid']):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(data_df.index, data_df['value'], color='black', linewidth=0.5, label='series')

        # Plot anomalies
        if meta_df is not None:
            date_index = data_df.index
            label = meta_df['label']
            value = data_df['value']
            if plot_vline:
                for xv in date_index[label == 1]:
                    ax.axvline(xv, color='orange', lw=1, alpha=0.1)
            ax.plot(date_index[label == 1], value[label == 1], linewidth=0, color='red', marker='x', markersize=5, label='anomalies')

        if title is not None:
            fig.suptitle(title, fontsize=18)

        legend = fig.legend(loc=0, prop={'size': 16})
        legend.get_frame().set_edgecolor('grey')
        legend.get_frame().set_linewidth(2.0)
    return fig


def plot_lag(data_df: pd.DataFrame, lag:int = 1, title: str=None):
    with plt.style.context(['seaborn-whitegrid']):
        ax = plotting.lag_plot(data_df['value'], lag=lag)

        if title is not None:
            ax.set_title(title)
