from pyadts.datasets import NABDataset
from pyadts.detectors import MatrixProfile


def test_matrix_profile():
    model = MatrixProfile(window_size=20)

    train_dataset = NABDataset(root='tests/data/nab', subset='realTweets', download=False)
    test_dataset = NABDataset(root='tests/data/nab', subset='realTweets', download=False)

    model.fit(train_dataset)

    scores = model.score(test_dataset)
    print(test_dataset.shape, scores.shape)
