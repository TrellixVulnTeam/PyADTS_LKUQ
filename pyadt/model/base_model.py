import abc


class BaseModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, X, y=None):
        pass

    @abc.abstractmethod
    def score(self, X):
        pass
