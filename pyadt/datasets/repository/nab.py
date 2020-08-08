import os
import json

import numpy as np
import pandas as pd

from ..series import Series


DATA_NAMES = {
    'nyc_taxi': 'data/realKnownCause/nyc_taxi.csv'
}

LABEL_NAME = 'labels/combined_windows.json'


def __check_dataset(root_path, dataset_name):
    if not os.path.exists(os.path.join(root_path, DATA_NAMES[dataset_name])):
        raise FileNotFoundError('The dataset %s not found in path %s' % (dataset_name, root_path))


def get_nab_nyc_taxi(root_path):
    __check_dataset(root_path, 'nyc_taxi')

    df = pd.read_csv(os.path.join(root_path, DATA_NAMES['nyc_taxi']))
    with open(os.path.join(root_path, LABEL_NAME)) as f:
        label_windows = json.load(f)['realKnownCause/nyc_taxi.csv']
    feature = df['value'].values
    timestamp = pd.to_datetime(df['timestamp'])
    label = np.zeros(len(df))

    for window in label_windows:
        t1 = pd.to_datetime(window[0])
        t2 = pd.to_datetime(window[1])

        label[(timestamp >= t1).values & (timestamp <= t2).values] = 1

    return Series(feature=feature, label=label, timestamp=timestamp)
