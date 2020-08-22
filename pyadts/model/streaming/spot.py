import numpy as np

from pyadts.model.base import BaseModel
from pyadts.utils.scaffold_algorithms import SPOT


class SPOT(BaseModel):
    def __init__(self, q: float= 1e-4, level: float= 0.02):
        super(SPOT, self).__init__()

        self.__spot = SPOT(q=q)
        self.__q = q
        self.__level = level

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        if x.ndim != 1:
            raise NotImplementedError('Only support 1-dimensional series!')

        self.store_train_data(x, y)
        self.__spot.initialize(level=self.__level, min_extrema=True)

    def predict_score(self, x: np.ndarray):
        self.check_fitted()
        self.__spot.fit(self.__train_x, x)
        result = self.__spot.run(dynamic=False)

        return result['alarms']
