import abc

import numpy as np


class Calibrator(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def calibrate(self, score: np.ndarrray):
        pass
