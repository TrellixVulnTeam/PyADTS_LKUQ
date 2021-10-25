"""
@Time    : 2021/10/25 15:24
@File    : score_ensembler.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from pyadts.generic import Function


class MaximumScoreEnsembler(Function):
    def __init__(self):
        super(MaximumScoreEnsembler, self).__init__()

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray):
        return np.max(x, axis=-1)

    def fit_transform(self, x: np.ndarray):
        return self.transform(x)


class AverageScoreEnsembler(Function):
    def __init__(self):
        super(AverageScoreEnsembler, self).__init__()

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray):
        assert x.ndim == 1
        return np.mean(x, axis=-1)

    def fit_transform(self, x: np.ndarray):
        return self.transform(x)


class MedianScoreEnsembler(Function):
    def __init__(self):
        super(MedianScoreEnsembler, self).__init__()

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray):
        assert x.ndim == 1
        return np.median(x, axis=-1)

    def fit_transform(self, x: np.ndarray):
        return self.transform(x)


class AOMScoreEnsembler(Function):
    def __init__(self, num_buckets: int = 5):
        super(AOMScoreEnsembler, self).__init__()

        self.num_buckets = num_buckets

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray):
        num_scores_per_bucket = x.shape[-1] // self.num_buckets
        shuffle_idx = np.arange(x.shape[-1])
        np.random.shuffle(shuffle_idx)

        candidates = []
        for st in range(0, x.shape[-1], num_scores_per_bucket):
            if st + num_scores_per_bucket > x.shape[-1]:
                bucket_scores = x[..., shuffle_idx][..., st:]
            else:
                bucket_scores = x[..., shuffle_idx][..., st: st + num_scores_per_bucket]

            candidates.append(np.max(bucket_scores, axis=-1))

        candidates = np.stack(candidates, axis=-1)
        return np.mean(candidates, axis=-1)

    def fit_transform(self, x: np.ndarray):
        pass


class MOAScoreEnsembler(Function):
    def __init__(self, num_buckets: int = 5):
        super(MOAScoreEnsembler, self).__init__()

        self.num_buckets = num_buckets

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray):
        num_scores_per_bucket = x.shape[-1] // self.num_buckets

    def fit_transform(self, x: np.ndarray):
        num_scores_per_bucket = x.shape[-1] // self.num_buckets
        shuffle_idx = np.arange(x.shape[-1])
        np.random.shuffle(shuffle_idx)

        candidates = []
        for st in range(0, x.shape[-1], num_scores_per_bucket):
            if st + num_scores_per_bucket > x.shape[-1]:
                bucket_scores = x[..., shuffle_idx][..., st:]
            else:
                bucket_scores = x[..., shuffle_idx][..., st: st + num_scores_per_bucket]

            candidates.append(np.mean(bucket_scores, axis=-1))

        candidates = np.stack(candidates, axis=-1)
        return np.max(candidates, axis=-1)
