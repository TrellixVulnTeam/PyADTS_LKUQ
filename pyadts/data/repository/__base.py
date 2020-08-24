import abc
from typing import Tuple

import pandas as pd


class get_dataset(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass
