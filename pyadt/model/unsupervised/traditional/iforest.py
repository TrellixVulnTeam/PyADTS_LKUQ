from sklearn.ensemble import IsolationForest

from ...base import BaseModel


class IForest(BaseModel):
    def __init__(self):
        super(IForest, self).__init__()
