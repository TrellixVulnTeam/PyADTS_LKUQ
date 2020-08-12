import sys

sys.path.append('..')

from pyadt.datasets.repository.kpi import get_kpi

from pyadt.datasets.utils import plot_series


def test_get_kpi():
    data = get_kpi('./data/kpi', id=0)
    fig = plot_series(value=data['value'], label=data['label'], datetime=data['datetime'])
    fig.show()
