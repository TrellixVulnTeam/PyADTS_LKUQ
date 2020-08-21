import numpy as np
from tqdm.std import tqdm


def get_window_feature(self, value: np.ndarray, window_size: int, verbose: bool = True):
    assert window_size * 2 <= value.shape[0]

    start_point = 2 * window_size
    # start_accum = 0
    data = []

    progress_bar = tqdm(np.arange(start_point, len(value)), desc='SLIDING_WINDOW') if verbose else np.arange(
        start_point, len(value))

    for i in progress_bar:
        # the datum to put into the data pool
        datum = []

        # fill the datum with features related to windows
        mean_w = np.mean(value[i - window_size:i + 1])
        var_w = np.mean((np.asarray(value[i - window_size:i + 1]) - mean_w) ** 2)
        # var_w = np.var(time_series[i-k:i+1])

        mean_w_and_1 = mean_w + (value[i - window_size - 1] - value[i]) / (window_size + 1)
        var_w_and_1 = np.mean((np.asarray(value[i - window_size - 1:i]) - mean_w_and_1) ** 2)
        # mean_w_and_1 = np.mean(time_series[i-k-1:i])
        # var_w_and_1 = np.var(time_series[i-k-1:i])

        mean_2w = np.mean(value[i - 2 * window_size:i - window_size + 1])
        var_2w = np.mean((np.asarray(value[i - 2 * window_size:i - window_size + 1]) - mean_2w) ** 2)
        # var_2w = np.var(time_series[i-2*k:i-k+1])

        # diff of sliding windows
        diff_mean_1 = mean_w - mean_w_and_1
        diff_var_1 = var_w - var_w_and_1

        # diff of jumping windows
        diff_mean_w = mean_w - mean_2w
        diff_var_w = var_w - var_2w

        # f1
        datum.append(mean_w)  # [0:2] is [0,1]
        # f2
        datum.append(var_w)
        # f3
        datum.append(diff_mean_1)
        # f4
        datum.append(diff_mean_1 / (mean_w_and_1 + 1e-10))
        # f5
        datum.append(diff_var_1)
        # f6
        datum.append(diff_var_1 / (var_w_and_1 + 1e-10))
        # f7
        datum.append(diff_mean_w)
        # f8
        datum.append(diff_mean_w / (mean_2w + 1e-10))
        # f9
        datum.append(diff_var_w)
        # f10
        datum.append(diff_var_w / (var_2w + 1e-10))

        # diff of sliding/jumping windows and current value
        # f11
        datum.append(value[i] - mean_w_and_1)
        # f12
        datum.append(value[i] - mean_2w)

        data.append(np.asarray(datum))

    return np.asarray(data)
