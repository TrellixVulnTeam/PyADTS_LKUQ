"""
@Time    : 2021/10/18 19:47
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import warnings
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import as_strided
from scipy import signal
from scipy.ndimage import gaussian_filter

from pyadts.generic import TimeSeriesDataset


def __autocorrelation(x):
    """
    Calculate the autocorrelation function of array ``x``.
    """
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]


def __interpolate_missing_data(times, fluxes, cadences=None):
    """
    Assuming ``times`` are uniformly spaced with missing cadences,
    fill in the missing cadences with linear interpolation.
    Cadences can be passed if they are known.
    Parameters
    ----------
    times : numpy.ndarray
        Incomplete but otherwise uniformly sampled times
    fluxes : numpy.ndarray
        Flux for each time in ``times``
    cadences : numpy.ndarray, optional
        Integer cadence number of each observation.
    Returns
    -------
    interpolated_times : numpy.ndarray
        ``times`` with filled-in missing cadences
    interpolated_fluxes : numpy.ndarray
        ``fluxes`` with filled-in missing cadences
    """
    first_time = times[0]

    if cadences is not None:
        # Median time between cadences
        dt = np.median(np.diff(times) / np.diff(cadences))
        cadence_indices = cadences - cadences[0]
    else:
        # Find typical time between cadences:
        dt = np.median(np.diff(times))
        # Approximate the patchy grid of integer cadence indices,
        # i.e.: (0, 1, 3, 4, 5, 8, ...)
        cadence_indices = np.rint((times - first_time) / dt)

    # Find missing cadence indices if that grid were complete
    expected_cadence_indices = set(np.arange(cadence_indices.min(),
                                             cadence_indices.max()))
    missing_cadence_indices = expected_cadence_indices.difference(set(cadence_indices))
    # Convert the missing cadences to times
    missing_times = first_time + np.array(list(missing_cadence_indices)) * dt

    # Interpolate to find fluxes at missing times
    interp_fluxes = np.interp(missing_times, times, fluxes)

    # Combine the interpolated and input times, fluxes
    interpolated_fluxes = np.concatenate([fluxes, interp_fluxes])
    interpolated_times = np.concatenate([times, missing_times])

    # Sort the times, fluxes, so that you can compute the ACF on them:
    sort_by_time = np.argsort(interpolated_times)
    interpolated_fluxes = interpolated_fluxes[sort_by_time]
    interpolated_times = interpolated_times[sort_by_time]
    return interpolated_times, interpolated_fluxes


def __interpolated_acf(times, fluxes, cadences=None):
    if not np.all(np.sort(times) == times):
        raise ValueError("Arrays must be in chronological order to compute ACF")

    if not np.abs(np.median(fluxes) / np.max(fluxes) < 0.01):
        warnmessage = ("Have you normalized your fluxes so that their median is"
                       " near zero?")
        warnings.warn(warnmessage)

    # Interpolate over missing times, fluxes
    interpolated_times, interpolated_fluxes = __interpolate_missing_data(times,
                                                                         fluxes,
                                                                         cadences)
    # Calculate the grid of "lags" in units of ``times``
    dt = np.median(np.diff(interpolated_times))
    lag = dt * np.arange(len(interpolated_fluxes))

    # Compute the autocorrelation function on interpolated fluxes
    acf = __autocorrelation(interpolated_fluxes)
    return lag, acf


def __dominant_period(lag, acf, min=None, max=None, fwhm=18, window=56,
                      plot=False, quiet=False):
    """
    Find the dominant period in the smoothed autocorrelation function.
    If no dominant period is found, raise `NoPeakWarning` and return `numpy.nan`
    Parameters
    ----------
    lag : numpy.ndarray
        Time lags
    acf : numpy.ndarray
        Autocorrelation function
    min : float (optional)
        Return dominant period greater than ``min``. Default is no limit.
    max : float (optional)
        Return dominant period less than ``max``. Default is no limit.
    fwhm : float (optional)
        Full-width at half max [lags] of the gaussian smoothing kernel. Default
        is 18 lags, as in McQuillan, Aigrain & Mazeh (2013) [1]_
    window : float (optional)
        Truncate the gaussian smoothing kernel after ``window`` lags. Default
        is 56 lags, as in McQuillan, Aigrain & Mazeh (2013) [1]_
    plot : bool (optional)
        Plot the autocorrelation function, peak detected. Default is `False`.
    quiet : bool (optional)
        Don't raise warning if no period is found. Default is `False`.
    Return
    ------
    acf_period : float
        Dominant period detected via the autocorrelation function
    References
    ----------
    .. [1] http://adsabs.harvard.edu/abs/2013MNRAS.432.1203M
    """
    lag_limited = np.copy(lag)
    acf_limited = np.copy(acf)

    # Apply limits if any are input:
    if min is not None and max is None:
        acf_limited = acf_limited[lag_limited > min]
        lag_limited = lag_limited[lag_limited > min]
    elif max is not None and min is None:
        acf_limited = acf_limited[lag_limited < max]
        lag_limited = lag_limited[lag_limited < max]
    elif max is not None and min is not None:
        acf_limited = acf_limited[(lag_limited < max) & (lag_limited > min)]
        lag_limited = lag_limited[(lag_limited < max) & (lag_limited > min)]

    # Convert fwhm -> sigma, convolve with gaussian kernel
    sigma = fwhm / 2.355
    truncate = window / sigma
    smooth_acf = gaussian_filter(acf_limited, sigma, truncate=truncate)

    # Detect peaks
    relative_maxes = signal.argrelmax(smooth_acf)[0]
    if len(relative_maxes) == 0:
        if not quiet:
            warnmsg = ("No period found. Be sure to check limits and to "
                       "median-subtract your fluxes.")
            warnings.warn(warnmsg)
        return np.nan

    # Detect highest peak
    absolute_max_index = relative_maxes[np.argmax(smooth_acf[relative_maxes])]
    acf_period = lag_limited[absolute_max_index]

    return acf_period


def sliding_window(x: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Fast implementation of the sliding window operation applied on the last dimension of `x`.

    >>> x = np.random.randn(16, 10, 2, 1000)  ## shape: (16, 10, 2, 1000)
    >>> window_size, stride = 100, 2
    >>> x_window = sliding_window(x, window_size, stride)
    >>> print(x_window.shape)  ## shape: (16, 10, 2, 451, 100)

    Args:
        x (np.ndarray): input time-series with shape (num_series, *, channel, timestamps)
        window_size ():
        stride ():
        copy ():

    Returns:
        res (np.ndarray): sliding windows with shape (num_series, *, channel, num_windows, window_size)
    """

    overlap = window_size - stride
    num_windows = (x.shape[-1] - overlap) // stride
    res_shape = (*x.shape[:-1], num_windows, window_size)
    res_strides = (
    *(np.array(x.strides[:-1]) * num_windows * window_size).tolist(), x.strides[-1] * stride, x.strides[-1])

    res = as_strided(x, shape=res_shape, strides=res_strides, writeable=False)

    return res


def auto_period(value: np.ndarray):
    lag, acf = __interpolated_acf(np.arange(value.shape[0]), value)
    period = __dominant_period(lag, acf, plot=False)

    return int(period)


# def label_sampling(x: np.ndarray, rate: float = 1.0, method: str = 'segment'):
#     rate = float(rate)
#     assert 0.0 <= rate <= 1.0
#
#     if method == 'segment':
#         if rate == 1.0:
#             return self
#         elif rate == 0.0:
#             return Series(value=self.value, timestamp=self.timestamp, label=None, name=self.name,
#                           normalized=self.normalized)
#         else:
#             anomalies_num = np.count_nonzero(self.label) * rate
#             sampled_label = np.copy(self.label).astype(np.int)
#             start = np.where(np.diff(sampled_label) == 1)[0] + 1  # Heads of anomaly segments
#             if sampled_label[0] == 1:
#                 start = np.concatenate([[0], start])
#             end = np.where(np.diff(sampled_label) == -1)[0] + 1  # Tails of anomaly segments
#             if sampled_label[-1] == 1:
#                 end = np.concatenate([end, [len(sampled_label)]])
#
#             segments = np.arange(len(start))  # Segment ids
#             np.random.shuffle(segments)
#
#             # Iterate segments
#             for i in range(len(start)):
#                 idx = (np.where(segments == i)[0]).item()
#                 sampled_label[start[idx]:end[idx]] = 0
#                 if np.count_nonzero(sampled_label) <= anomalies_num:
#                     break
#
#             return Series(value=self.value, timestamp=self.timestamp, label=sampled_label, name=self.name,
#                           normalized=self.normalized)
#     elif method == 'point':
#         if rate == 1.0:
#             return self
#         elif rate == 0.0:
#             return Series(value=self.value, timestamp=self.timestamp, label=None, name=self.name,
#                           normalized=self.normalized)
#         else:
#             anomaly_indices = np.arange(self.length)[self.label == 1]
#             selected_indices = np.random.choice(anomaly_indices,
#                                                 size=int(np.floor(anomaly_indices.shape[0] * (1 - rate))),
#                                                 replace=False)
#             sampled_label = np.copy(self.label).astype(np.int)
#             sampled_label[selected_indices] = 0
#
#             return Series(value=self.value, timestamp=self.timestamp, label=sampled_label, name=self.name,
#                           normalized=self.normalized)
#     else:
#         raise ValueError('Invalid label sampling method!')


def timestamp_to_datetime(ts: Union[int, float]) -> datetime:
    return datetime.fromtimestamp(ts if isinstance(ts, int) else int(ts))


def datetime_to_timestamp(dt: datetime) -> int:
    return int(dt.timestamp())


def validate_timeseries():
    pass


def rearrange_dataframe(df: pd.DataFrame, time_col: str = None, sort_by_time: bool = True, resampling: bool = True,
                        tackle_missing: str = 'ffill'):
    if sort_by_time or resampling:
        assert time_col is not None

    res_df = df.copy(deep=True)

    if resampling:
        if tackle_missing is None or tackle_missing == 'none':
            warnings.warn(
                'Resampling time-series may result in missing values. Setting `tackle_missing` is recommended.')
        time_series = res_df[time_col]
        if isinstance(time_series.iloc[0], int):
            time_series = time_series.apply(timestamp_to_datetime)
        datetime_series = pd.to_datetime(time_series)
        res_df.set_index(datetime_series, inplace=True)
        res_df.drop(columns=[time_col], inplace=True)
        min_time_interval = datetime_series.diff().min()
        res_df = res_df.resample(min_time_interval).asfreq()
        new_datetime_series = res_df.index
        res_df.reset_index(inplace=True)
        res_df[time_col] = new_datetime_series

    if sort_by_time:
        res_df = res_df.sort_values(by=time_col)

    if tackle_missing == 'ffill':
        res_df = res_df.fillna(method='ffill')
    elif tackle_missing == 'bfill':
        res_df = res_df.fillna(method='bfill')
    elif tackle_missing == 'fzero':
        res_df = res_df.fillna(0)
    elif tackle_missing == 'drop':
        res_df = res_df.dropna(axis=0)
    elif tackle_missing is None or tackle_missing == 'none':
        pass
    else:
        raise ValueError

    return res_df


def any_to_numpy(x: Union[list, TimeSeriesDataset, np.ndarray, torch.Tensor]):
    if isinstance(x, np.ndarray):
        pass
    elif isinstance(x, list):
        return np.asarray(x)
    elif isinstance(x, torch.Tensor):
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()
    elif isinstance(x, TimeSeriesDataset):
        return x.to_numpy()
    else:
        raise ValueError


def any_to_tensor(x: Union[list, TimeSeriesDataset, np.ndarray, torch.Tensor], device='cpu'):
    if isinstance(x, np.ndarray):
        out = torch.from_numpy(x)
    elif isinstance(x, list):
        out = torch.from_numpy(np.asarray(x))
    elif isinstance(x, torch.Tensor):
        out = x
    elif isinstance(x, TimeSeriesDataset):
        out = x.to_tensor()
    else:
        raise ValueError

    out = out.to(device)
    return out
