import numpy as np

from pyadts.generic import Calibrator


class ThresholdCalibrator(Calibrator):
    def __init__(self, threshold: float = 0.5):
        super(ThresholdCalibrator, self).__init__()

        self.threshold = threshold

    def calibrate(self, score: np.ndarrray):
        result = np.zeros_like(score)
        result[score > self.threshold] = 1

        return result
