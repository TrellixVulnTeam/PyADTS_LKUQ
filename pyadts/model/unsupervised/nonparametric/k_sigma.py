import numpy as np

from ...base import BaseModel


class KSigmaDetector(BaseModel):
    def __init__(self):
        super(KSigmaDetector, self).__init__()

    def fit(self, X: np.ndarray, y: np.ndarray=None):
        pass

    def predict_score(self, X: np.ndarray):
        pass

    def predict(self, X: np.ndarray):
        assert self.__check_fitted()
