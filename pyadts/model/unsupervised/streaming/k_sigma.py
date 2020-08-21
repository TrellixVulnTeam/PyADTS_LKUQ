import numpy as np

from pyadts.model.base import BaseModel


class KSigmaDetector(BaseModel):
    def __init__(self):
        super(KSigmaDetector, self).__init__()

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        pass

    def predict_score(self, x: np.ndarray):
        pass

    def predict(self, x: np.ndarray):
        assert self.check_fitted()
