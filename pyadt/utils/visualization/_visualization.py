import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(series: np.ndarray, label: np.ndarray=None, timestamp: pd.DatetimeIndex = None, plot_vline=True):
    if timestamp is None:
        timestamp = np.arange(len(series))

    with plt.style.context(['time-series-grid']):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(timestamp, series, color='black', linewidth=0.5)

        # Plot anomalies
        if label is not None:
            if plot_vline:
                for xv in timestamp[label==1]:
                    ax.axvline(xv, color='orange', lw=1, alpha=0.1)
            ax.plot(timestamp[label == 1], series[label == 1], linewidth=0, color='red', marker='x', markersize=5, label='anomalies')
        fig.legend(loc=0, prop={'size': 18})
    return fig
