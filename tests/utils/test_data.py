"""
@Time    : 2021/11/11 10:59
@File    : test_data.py
@Software: PyCharm
@Desc    : 
"""
import pandas as pd

from pyadts.utils.data import rearrange_dataframe


def test_rearrange_dataframe():
    df = pd.read_csv('tests/data/kpi/phase2_train.csv')
    df = df[df['KPI ID'] == df.loc[0, 'KPI ID']]

    res_df = rearrange_dataframe(df, time_col='timestamp', tackle_missing=None)
    print(res_df)
