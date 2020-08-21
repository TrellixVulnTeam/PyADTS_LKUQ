import numpy as np


def get_log_feature(value: np.ndarray) -> np.ndarray:
    return np.log(value)


def get_diff_feature(value: np.ndarray) -> np.ndarray:
    return np.diff(value, prepend=0)


def get_diff2_feature(value: np.ndarray) -> np.ndarray:
    return np.diff(np.diff(value, prepend=0), prepend=0)
