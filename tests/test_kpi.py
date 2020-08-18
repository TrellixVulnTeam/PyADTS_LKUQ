import sys

import matplotlib.pyplot as plt

sys.path.append('..')

from pyadt.datasets.repository.kpi import get_kpi

from pyadt.utils.visualization import plot_series, plot_lag


def test_get_kpi():
    data_df, meta_df = get_kpi('./data/kpi', id=0)
    print(data_df)
    print(meta_df)
    fig = plot_series(data_df, meta_df, title='Vis')
    plt.show()

    plot_lag(data_df, 'test')
    plt.show()
