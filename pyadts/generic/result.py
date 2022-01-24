import abc
from typing import Union, List

import numpy as np


class DetectionResult(abc.ABC):
    def __init__(self, scores: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(scores, np.ndarray):
            self.scores = [scores]
        else:
            self.scores = scores

        self.predictions = None

    def __repr__(self):
        pass
