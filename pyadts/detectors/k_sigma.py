import numpy as np

from pyadts.generic import Model


class KSigmaDetector(Model):
    def __init__(self):
        super(KSigmaDetector, self).__init__()

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        pass

    def predict_score(self, x: np.ndarray):
        pass

    def predict(self, x: np.ndarray):
        pass
