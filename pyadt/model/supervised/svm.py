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

    def fit(self, X: np.ndarray, y: np.ndarray=None):
        self.__store_train_data(X, y)

        self.model.fit(X, y)

    def predict_score(self, X: np.ndarray):
        self.__check_fitted()

        scores = self.model.decision_function(X)

        # TODO
        return scores[:,1].reshape(-1)

    def predict(self, X: np.ndarray):
        assert self.__check_fitted()

        return self.model.predict(X)
