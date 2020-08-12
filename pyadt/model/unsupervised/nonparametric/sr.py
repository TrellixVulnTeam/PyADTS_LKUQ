from enum import Enum

import numpy as np
import pandas as pd

from ...base import BaseModel


class SpectralResidual(BaseModel):
    def __init__(self):
        super(SpectralResidual, self).__init__()
