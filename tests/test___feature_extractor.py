import sys

sys.path.append('../')

from pyadts.data.repository.kpi import get_kpi
from pyadts.data.preprocessing import (
    series_rearrange, series_normalize, series_impute
)
from pyadts.data.feature import (
    get_all_features
)


def test_extract_features():
    data_df, meta_df = get_kpi('./data/kpi', kpi_id=0)
    data_df, meta_df = series_rearrange(data_df, meta_df)
    series_impute(data_df, method='linear')
    series_normalize(data_df, method='minmax')

    # sr_feature = get_sr_feature(x)
    # log_feature = get_log_feature(x)
    # diff_feature = get_diff_feature(x)
    # diff2_feature = get_diff2_feature(x)
    # sarima_feature = get_sarima_feature(x)
    # addes_feature = get_addes_feature(x)
    # holt_feature = get_holt_feature(x)
    # simplees_feature = get_simplees_feature(x)
    # wavelet_feature = get_wavelet_feature(x)
    # stl_feature = get_stl_feature(x, period=1440)
    # window_feature = get_window_feature(x, window_size=100)

    features = get_all_features(data_df, get_stl=False, window_list=[10, 100])
