import sys

sys.path.append('..')

from pyadt.datasets.series import Series
from pyadt.datasets.repository.nab import get_nab_nyc_taxi

from pyadt.utils.visualization import plot


def test_get_nab_nyc_taxi():
    series = get_nab_nyc_taxi('./data/nab')
    # fig = plot(series.feature, series.label, series.timestamp)
    # fig.show()
