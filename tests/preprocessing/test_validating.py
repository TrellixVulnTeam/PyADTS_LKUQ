import numpy as np

from pyadts.datasets import KPIDataset
from pyadts.preprocessing import rearrange_timestamps

dataset = KPIDataset(root='tests/data/kpi', download=False)


def test_rearrange_timestamps():
    old_timestamps = dataset.timestamps
    new_dataset = rearrange_timestamps(dataset)
    new_timestamps = new_dataset.timestamps
    for ts in old_timestamps:
        print(np.unique(np.diff(np.sort(ts))))

    for ts in new_timestamps:
        print(np.unique(np.diff(ts)))


def test_fill_missing():
    pass
