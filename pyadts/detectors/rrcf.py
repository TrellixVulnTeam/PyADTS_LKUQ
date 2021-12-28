"""
@Time    : 2021/10/25 12:03
@File    : rrcf.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union

import numpy as np
from tqdm.std import tqdm

from pyadts.generic import Detector, TimeSeriesDataset
from pyadts.utils.data import any_to_numpy
from .__rrcf_base import RCTree


class RRCF(Detector):
    def __init__(self, num_trees: int = 20, max_leaves: int = 128):
        super(RRCF, self).__init__()

        self.num_trees = num_trees
        self.max_leaves = max_leaves
        self.train_size = 0

        self.forest = []

    def fit(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor], y=None):
        x = any_to_numpy(x)
        num_points = len(x)
        self.train_size = num_points

        for i_tree in range(self.num_trees):
            self.forest.append(RCTree())

        for i_point in tqdm(range(num_points), desc='::Training::'):
            for tree in self.forest:
                if len(tree.leaves) > self.max_leaves:
                    tree.forget_point(i_point - self.max_leaves)
                tree.insert_point(x[i_point], index=i_point)

    def score(self, x: Union[TimeSeriesDataset, np.ndarray, torch.Tensor]):
        x = any_to_numpy(x)
        num_points = len(x)
        scores = np.zeros(num_points)

        for i_point in tqdm(range(num_points), desc='::Evaluation::'):
            for tree in self.forest:
                if len(tree.leaves) > self.max_leaves:
                    tree.forget_point(i_point - self.max_leaves)
                tree.insert_point(x[i_point], index=self.train_size + i_point)
                scores[i_point] += (tree.codisp(i_point + self.train_size) / self.num_trees)

        return scores
