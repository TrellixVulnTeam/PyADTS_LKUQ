"""
@Time    : 2021/10/18 0:43
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from ...utils.scaffold_algorithms import SPOT


def auto_threshold(train_score, score, q=1e-3, level=0.02):
    s = SPOT(q)
    s.fit(train_score, score)
    s.initialize(level=level, min_extrema=True)
    result = s.run(dynamic=False)
    threshold = -np.mean(result['thresholds'])

    return threshold
