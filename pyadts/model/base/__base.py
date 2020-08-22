import abc

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pyadts.model.utils.__auto_threshold import auto_threshold


class BaseModel(abc.ABC):
    """
    The base model of anomaly detectors.
    """

    def __init__(self):
        pass

    def store_train_data(self, x: np.ndarray, y: np.ndarray = None):
        self.train_x = x
        self.train_y = y

    def check_fitted(self):
        return hasattr(self, 'train_x')

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray = None):
        """


        Parameters
        ----------
        x :
        y :
        """
        pass

    @abc.abstractmethod
    def predict_score(self, x: np.ndarray):
        """

        Parameters
        ----------
        x :
        """
        pass

    def set_params(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
        """
        self.__dict__.update(kwargs)

    def get_params(self):
        """

        Returns
        -------

        """
        return self.__dict__

    def predict(self, x: np.ndarray):
        """

        Parameters
        ----------
        x :

        Returns
        -------

        """
        assert self.check_fitted()

        train_score = self.predict_score(self.__train_x)
        test_score = self.predict_score(x)

        th = auto_threshold(train_score, test_score)

        predictions = np.zeros_like(test_score)
        predictions[test_score < th] = 1

        return predictions.astype(np.int)

    def predict_prob(self, x: np.ndarray):
        """

        Parameters
        ----------
        x :

        Returns
        -------

        """
        assert self.check_fitted()

        train_score = self.predict_score(self.__train_x)
        test_score = self.predict_score(x)

        scaler = MinMaxScaler().fit(train_score.reshape(-1, 1))
        prob = scaler.transform(test_score.reshape(-1, 1)).ravel().clip(0, 1).reshape(-1)

        return prob

    def fit_predict(self, x: np.ndarray, y: np.ndarray = None):
        """

        Parameters
        ----------
        x :
        y :

        Returns
        -------

        """
        self.fit(x, y)

        return self.predict(x)

    def fit_predict_score(self, x: np.ndarray, y: np.ndarray = None):
        """

        Parameters
        ----------
        x :
        y :

        Returns
        -------

        """
        self.fit(x, y)

        return self.predict_score(x)

    def fit_predict_prob(self, x: np.ndarray, y: np.ndarray = None):
        """

        Parameters
        ----------
        x :
        y :

        Returns
        -------

        """
        self.fit(x, y)

        return self.predict_prob(x)
