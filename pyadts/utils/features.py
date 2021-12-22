from typing import Union, List

import numpy as np
from scipy import signal


def get_min_feature(x: np.ndarray):
    """
        Getting the minimum value of time series array.

        Args:
            x:

        Returns:
            The minimum value of the time series array.
        """
    return np.min(x, axis=-1, keepdims=True)


def get_max_feature(x: np.ndarray):
    """
        Getting the maximum value of time series array.

        Args:
            x:

        Returns:
            The maximum of the time series array.
        """
    return np.max(x, axis=-1, keepdims=True)


def get_mean_feature(x: np.ndarray):
    """
    Getting the average value of time series array.

    Args:
        x:

    Returns:
        Average of the time series array.
    """
    return x.mean(axis=-1, keepdims=True)


def get_var_feature(x: np.ndarray):
    """
    Getting the variance of time series array.

    Args:
        x:

    Returns:
        Variance of the time series array.
    """
    return x.var(axis=-1, keepdims=True)


def get_diff_mean_feature(x: np.ndarray):
    """
        Getting the average value of time series difference array.

        Args:
            x:

        Returns:
            Average of the time series difference array.
        """
    return np.diff(x, axis=-1).mean(axis=-1, keepdims=True)


def get_diff_var_feature(x: np.ndarray):
    """
        Getting the variance of time series difference array.

        Args:
            x:

        Returns:
            Variance of the time series difference array.
        """
    return np.diff(x, axis=-1).var(axis=-1, keepdims=True)


def get_spectral_entropy(x: np.ndarray, freq: int = 1):
    """
    Getting normalized Shannon entropy of power spectral density. PSD is calculated using scipy's periodogram.

    Args:
        x:
        freq:

    Returns:
        Normalized Shannon entropy.
    """
    _, psd = signal.periodogram(x, freq, axis=-1)

    # calculate shannon entropy of normalized psd
    psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
    entropy = np.nansum(psd_norm * np.log2(psd_norm), axis=-1, keepdims=True)

    return -(entropy / np.log2(psd_norm.size))


def get_lumpiness(x: np.ndarray, window_size: int = 20):
    """
    Calculating the lumpiness of time series.
    Lumpiness is defined as the variance of the chunk-wise variances.

    Args:
        x: The time series array.
        window_size: int; Window size to split the data into chunks for getting variances. Default value is 20.

    Returns:
        Lumpiness of the time series array.
    """
    v = np.stack([np.var(x_w, axis=-1) for x_w in np.array_split(x, x.shape[-1] // window_size + 1, axis=-1)], axis=-1)
    return np.var(v, axis=-1)


def get_stability(x: np.ndarray, window_size: int = 20):
    """
    Calculating the stability of time series.
    Stability is defined as the variance of chunk-wise means.

    Args:
        x: The time series array.
        window_size: int; Window size to split the data into chunks for getting variances. Default value is 20.

    Returns:
        Stability of the time series array.
    """
    v = np.stack([np.mean(x_w, axis=-1) for x_w in np.array_split(x, x.shape[-1] // window_size + 1, axis=-1)], axis=-1)
    return np.var(v, axis=-1)


def get_stl_features(x: np.ndarray):
    pass


def get_level_shift(x: np.ndarray):
    pass


def get_flat_spots(x: np.ndarray):
    pass


def get_hurst_exponent(x: np.ndarray):
    pass


def get_acf_feature(x: np.ndarray):
    pass


def get_pacf_feature(x: np.ndarray):
    pass


def get_crossing_points(x: np.ndarray):
    pass


def get_unitroot_kpss(x: np.ndarray):
    pass


def get_heterogeneity(x: np.ndarray):
    pass


class FeatureExtractor(object):
    registered_features = {
        'min': get_min_feature,
        'max': get_max_feature,
        'mean': get_mean_feature,
        'var': get_var_feature,
        'diff_mean': get_diff_mean_feature,
        'diff_var': get_diff_var_feature,
        'spectral_entropy': get_spectral_entropy,
        'lumpiness': get_lumpiness,
        'stability': get_stability
    }
    available_features = list(registered_features.keys())

    def __init__(self, features: Union[str, List[str]] = 'all'):
        if features == 'all':
            selected_features = self.available_features
        elif isinstance(features, str):
            assert features in self.available_features, f'The feature `{features}` is not available or not valid!'
            selected_features = [features]
        elif isinstance(features, list):
            for feature in features:
                assert feature in self.available_features, f'The feature `{feature}` is not available or not valid!'
            selected_features = features
        else:
            raise ValueError

        self.selected_features = selected_features

    def __call__(self, x: np.ndarray):
        pass
