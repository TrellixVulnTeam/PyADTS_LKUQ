import sys

from pyadts.data.repository import get_kpi
from pyadts.data.preprocessing import series_normalize

sys.path.append('..')


def test_series_normalize():
    data_df, meta_df = get_kpi('./data/kpi', kpi_id=0)
    series_normalize(data_df, method='minmax')
    series_normalize(data_df, method='negpos1')
    series_normalize(data_df, method='zscore')
