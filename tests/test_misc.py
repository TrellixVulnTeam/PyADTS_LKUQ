"""
@Time    : 2021/10/18 20:08
@File    : test_misc.py
@Software: PyCharm
@Desc    : 
"""

import numpy as np

from pyadts.utils import sliding_window_with_stride


def test_misc():
    # x = np.random.randn(16, 1000)  ## shape: (16, 10, 2, 1000)
    # window_size, stride = 100, 2
    # x_window = sliding_window_with_stride(x, window_size, stride)
    # x_window_true = np.zeros((*x.shape[:-1], (x.shape[-1] - (window_size - stride)) // stride, window_size))
    # for i, j, k in itertools.product(range(x.shape[0]), range(x.shape[1]), range(x.shape[2])):
    #     for ts in range((x.shape[-1] - (window_size - stride)) // stride):
    #         x_window_true[i, j, k, ts] = x[i, j, k, ts]

    x = np.arange(100)
    window_size, stride = 10, 2
    x_window = sliding_window_with_stride(x, window_size, stride)
    x_window_true = []
    for i in range(0, 100, 2):
        if i + window_size > 100:
            break
        x_window_true.append(x[i:i + window_size])

    x_window_true = np.stack(x_window_true, axis=0)

    print(x_window.shape)
    print(x_window_true.shape)

    assert (x_window == x_window_true).all()
