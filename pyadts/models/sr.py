import numpy as np

from pyadts.common import Model


class SpectralResidual(Model):
    def __init__(self):
        super(SpectralResidual, self).__init__()

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        if x.ndim != 1:
            raise NotImplementedError('Only support 1-dimensional series!')

        self.store_train_data(x, y)

    def predict(self):
        pass

    def score(self, x: np.ndarray):
        self.check_fitted()

        mags = spectral_residual_transform(np.concatenate([self.train_x, x]))
        score = generate_spectral_score(mags)

        return score[-x.shape[0]:]
