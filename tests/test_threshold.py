import sys

sys.path.append('..')

from pyadts.data.repository.kpi import get_kpi
from pyadts.data.utils import train_test_split
from pyadts.model.streaming import ThresholdDetector
from pyadts.data.preprocessing import (
    series_rearrange, series_impute
)


def test_fit():
    data_df, meta_df = get_kpi('./data/kpi', kpi_id=0)
    data_df, meta_df = series_rearrange(data_df, meta_df)
    # series_normalize(data_df, method='zscore')
    series_impute(data_df, method='linear')

    train_data, train_meta, test_data, test_meta = train_test_split(data_df, meta_df, train_ratio=0.7)

    detector = ThresholdDetector(high=2, low=1)

    detector.fit(train_data.values)


def test_predict_score():
    data_df, meta_df = get_kpi('./data/kpi', kpi_id=0)
    data_df, meta_df = series_rearrange(data_df, meta_df)
    # series_normalize(data_df, method='zscore')
    series_impute(data_df, method='linear')

    train_data, train_meta, test_data, test_meta = train_test_split(data_df, meta_df, train_ratio=0.7)

    detector = ThresholdDetector(high=2, low=1)

    detector.fit(train_data.values)

    scores = detector.predict_score(test_data.values)


def test_predict():
    data_df, meta_df = get_kpi('./data/kpi', kpi_id=0)
    data_df, meta_df = series_rearrange(data_df, meta_df)
    # series_normalize(data_df, method='zscore')
    series_impute(data_df, method='linear')

    train_data, train_meta, test_data, test_meta = train_test_split(data_df, meta_df, train_ratio=0.7)

    detector = ThresholdDetector(high=2, low=1)

    detector.fit(train_data.values)

    predictions = detector.predict(test_data.values)
