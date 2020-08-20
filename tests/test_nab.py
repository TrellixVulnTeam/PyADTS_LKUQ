import sys

sys.path.append('..')

from pyadt.data.repository.nab import get_nab_nyc_taxi


def test_get_nab_nyc_taxi():
    series = get_nab_nyc_taxi('./data/nab')
    series.plot_series()
