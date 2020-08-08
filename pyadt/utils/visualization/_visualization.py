import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(series: np.ndarray, label: np.ndarray=None, timestamp: pd.DatetimeIndex = None):
    if timestamp is None:
        timestamp = np.arange(len(series))

    with plt.style.context(['time-series-grid']):
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(timestamp, series, color='black', linewidth=0.5)
        if label is not None:
            ax.plot(timestamp[label==1], series[label==1], linewidth=0, color='red', marker='o')

    return fig
