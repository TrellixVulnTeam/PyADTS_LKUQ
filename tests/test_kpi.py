import sys
from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')

from pyadts.data.repository import get_kpi

from pyadts.utils.visualization import plot_series


def test_get_kpi():
    data_df, meta_df = get_kpi('./data/kpi', kpi_id=0)
    print(data_df)
    print(meta_df)
    assert 'value' in data_df.columns and data_df.shape[1] == 1
    assert 'label' in meta_df.columns and 'timestamp' in meta_df.columns and meta_df.shape[1] == 2

    data_df['value2'] = np.random.randn(data_df.shape[0])
    fig = plot_series(data_df=data_df, meta_df=meta_df, title='vis')
    fig.show()
