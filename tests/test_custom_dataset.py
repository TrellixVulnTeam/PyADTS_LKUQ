"""
@Time    : 2020/9/13 13:59
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : test_custom_dataset.py
@Software: PyCharm
@Desc    : 
"""
import sys

sys.path.append('../')

from pyadts.data.repository import get_custom_dataset


def test_get_custom_dataset():
    data_df, meta_df = get_custom_dataset(
        root_path=r'data/travelsky/hangxing_check_STL_3Sigma_AV@baseline@10.5.72.5@AVCOMM@tps_average@201909.csv')
    print(data_df.head())
    print(meta_df.head())
