import numpy as np
from sklearn.svm import SVC, LinearSVC

from ..base import BaseModel


class SVM(BaseModel):
    def __init__(self, kernel='linear'):
        super().__init__()

        if kernel == 'linear':
            self.model = LinearSVC()
        else:
            self.model = SVC(kernel=kernel)

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        self.store_train_data(x, y)

        self.model.fit(x, y)

    def predict_score(self, x: np.ndarray):
        self.check_fitted()

        scores = self.model.decision_function(x)

        # TODO
        return scores[:, 1].reshape(-1)

    def predict(self, x: np.ndarray):
        assert self.check_fitted()

        return self.model.predict(x)
