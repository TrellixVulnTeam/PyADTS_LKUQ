import sys

import matplotlib.pyplot as plt

sys.path.append('..')

from pyadts.data.repository import get_kpi

from pyadts.utils.visualization import plot_series


def test_get_kpi():
    data_df, meta_df = get_kpi('./data/kpi', kpi_id=0)
    print(data_df)
    print(meta_df)
    fig = plot_series(data_df, meta_df, title='Vis')
    plt.show()
