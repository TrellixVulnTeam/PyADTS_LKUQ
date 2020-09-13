"""
@Time    : 2020/9/13 13:57
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : custom_dataset.py
@Software: PyCharm
@Desc    : 
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd

from ..utils import timestamp_to_datetime


def get_custom_dataset(root_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert os.path.splitext(root_path)[-1] == '.csv', 'Only .csv files are supported!'

    df = pd.read_csv(root_path)
    timestamp = df['timestamp'].values
    timeindex = pd.to_datetime(df['timestamp'])
    value = df['value'].values
    label = df['label'].values.astype(np.int)

    data_df = pd.DataFrame({'value': value}, index=timeindex)
    meta_df = pd.DataFrame({'label': label, 'timestamp': timestamp}, index=timeindex)

    return data_df, meta_df
