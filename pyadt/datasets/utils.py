import warnings
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime


def timestamp_to_datetime(ts: Union[int, float]) -> datetime:
    return datetime.fromtimestamp(ts if isinstance(ts, int) else int(ts))


def datetime_to_timestamp(dt: datetime) -> int:
    return int(dt.timestamp())


def handle_zeros(value):
    EPS = 1e-6
    if value < EPS:
        warnings.warn(f'Zero denominator detected. Replace it by {EPS}.')
        return EPS
    return value


def __missing_num(missing):
    return np.count_nonzero(missing)


def __missing_rate(missing):
    return np.count_nonzero(missing) / missing.shape[0]


def __anomaly_num(label):
    return np.count_nonzero(label)


def __anomaly_rate(label):
    return np.count_nonzero(label)/label.shape[0]


def dataset_statics(missing: np.ndarray, label: np.ndarray):
    return {'Missing num': __missing_num(missing), 'Missing rate': __missing_rate(missing),
            'Anomaly num': __anomaly_num(label), 'Anomaly rate': __anomaly_rate(label)}


def plot(series: np.ndarray, label: np.ndarray=None, timestamp: pd.DatetimeIndex = None, plot_vline=True):
    if timestamp is None:
        timestamp = np.arange(len(series))

    with plt.style.context(['ggplot']):
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
