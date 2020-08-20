import numpy as np


def average(scores: np.ndarray, weights=None):
    """The average combination method to combine the scores from different anomaly
    detectors by taking the average.
    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_detectors)
        The score matrix of different anomaly detectors.
    weights : numpy array of shape (1, n_detectors)
        Weighted for different detectors.
    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined scores.
    """

    pass


def maximization(scores: np.ndarray):
    """The maximization combination method to combine the scores from different
    anomaly detectors by taking the maximization.
    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_detectors)
        The score matrix of different anomaly detectors.
    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined scores.
    """

    pass


def average_of_maximum(scores):
    """
    """

    pass


def maximum_of_average(scores):
    """
    """

    pass


def majority_vote(scores, n_classes=2, weights=None):
    """
    """

    pass
