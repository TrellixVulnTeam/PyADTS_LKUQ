import sys

sys.path.append('..')

from pyadt.datasets.series import Series
from pyadt.datasets.repository.kpi import get_kpi

from pyadt.utils.visualization import plot


def test_get_kpi():
    series = get_kpi('./data/kpi', id=0)
    series.plot()
