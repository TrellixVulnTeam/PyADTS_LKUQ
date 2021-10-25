from pyadts.generic import Model


class AutoEncoder(Model):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError
