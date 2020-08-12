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


def train_test_split():
    # TODO
    pass


def label_sampling():
    # TODO
    pass


def sliding_window():
    # TODO
    pass


def plot_figure(value: np.ndarray, label: np.ndarray=None, datetime: Union[pd.DatetimeIndex, pd.Series] = None, plot_vline=True):
    if datetime is None:
        datetime = np.arange(len(value))

    with plt.style.context(['ggplot']):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot_figure(datetime, value, color='black', linewidth=0.5)

        # Plot anomalies
        if label is not None:
            if plot_vline:
                for xv in datetime[label == 1]:
                    ax.axvline(xv, color='orange', lw=1, alpha=0.1)
            ax.plot_figure(datetime[label == 1], value[label == 1], linewidth=0, color='red', marker='x', markersize=5, label='anomalies')
        fig.legend(loc=0, prop={'size': 18})
    return fig
