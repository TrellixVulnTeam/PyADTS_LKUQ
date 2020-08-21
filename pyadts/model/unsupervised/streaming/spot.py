import numpy as np

from pyadts.model.base import BaseModel
from pyadts.utils.scaffold_algorithms import SPOT


class SPOT(BaseModel):
    def __init__(self):
        super(SPOT, self).__init__()

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        pass

    def predict_score(self, x: np.ndarray):
        pass
