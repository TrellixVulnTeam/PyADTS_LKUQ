"""
@Time    : 2021/10/26 0:03
@File    : skab.py
@Software: PyCharm
@Desc    : 
"""
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import check_existence


class SKABDataset(TimeSeriesDataset):
    __categories = ['valve1', 'valve2', 'other']
    __file_list = {
        'valve1': {
            '0.csv': 'be2ca3c4e52b5c80167fb82c1a3d78b8', '1.csv': 'ce14ac9dbd5edc67918bbf5746ba0fd8',
            '10.csv': '7463097764e161b0dd63ecba8b39688b', '11.csv': 'd31e10cbc9c84932b917777d60d0eec5',
            '12.csv': 'a7b52ca8133d74ac3af701655bcbb9fc', '13.csv': '97493427af37f2464ca13f4a95dd83dd',
            '14.csv': '46e3195763bcddd43cdea00ed3182b09', '15.csv': 'd579c9221646a29bb5c055750d634ff4',
            '2.csv': '8844b8edb13c4233fd3e63e66227018d', '3.csv': '268833389d5c039e5bae6452ee4f1c72',
            '4.csv': '6d92507567ce32a4a9bc7841eac98e4a', '5.csv': '3abb2674ddf45eac917fdbde8eef7e27',
            '6.csv': 'fddad34dbf89e53661ef1ed92c33e7ed', '7.csv': '41c224ea408fa31c1dc564621c10b331',
            '8.csv': '47346bb9da43cba8ef801001ef506e32', '9.csv': '19139543237dc1cb0ecc042ee73bc595'
        },
        'valve2': {
            '0.csv': '4ea99b4fba5ceb3007ed893c9d48e04d', '1.csv': 'b723608b1c0724c03aaff9260abb44ea',
            '2.csv': '54014a3760477b90b3d7bc428bd6f500', '3.csv': '9bd8c23c64e7d4f2683f87a2bf36c740'
        },
        'other': {
            '11.csv': 'eebd9968f937f842867af0f128bcc7e8', '12.csv': '4e0350498ceb2f14badee00d039ebd66',
            '13.csv': '806b952ca4581f7df0253012629de89d', '14.csv': '07295494d162d9b073567dbd0a859e4f',
            '15.csv': '61e8723ec1ed612325904455b8c0592e', '16.csv': 'ff73f9e7f6278aed69a87ef2548304a9',
            '17.csv': 'ed53d2b66153bb385cc76c91ccf13136', '18.csv': '3976d9c3b72d7b3dae907eb568b260ef',
            '19.csv': '5ae6ee99a47623819495b57a479265d8', '20.csv': '51c5ff35ea11f2029cbf9e7d4acb81dd',
            '21.csv': 'afca8e4989e6bf503f572aa992937e03', '22.csv': '19f1f586352ad1b70b9e4abaf57502d9',
            '23.csv': 'd1544226cd34d79a411f646d481c7bd6', '9.csv': 'a5bc844a2247deb1697319f52dff1880'
        }
    }

    def __init__(self, root: str = None, category: str = None, download: bool = False):
        super(SKABDataset, self).__init__()
        assert category in self.__categories

        if root is None:
            root_path = Path.home() / 'skab'
            warnings.warn(
                f'The `root` path of the dataset is not set, using user home dir {str(root_path)} as default.')
        else:
            root_path = Path(root)

        if download:
            raise ValueError('The SMD dataset should be downloaded manually. '
                             'Please download the dataset at `https://github.com/waico/SKAB`!')
        else:
            self.__check_integrity(root_path)

        self.data = []
        self.labels = []

        for csv_file in (root_path / category).glob('*.csv'):
            df = pd.read_csv(csv_file, delimiter=';')
            df.sort_values(by='datetime', inplace=True)
            label = np.logical_or(df['anomaly'].values, df['changepoint'].values).astype(np.long)
            df.drop(columns=['datetime', 'anomaly', 'changepoint'], inplace=True)
            self.data.append(df.values)
            self.labels.append(label)

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        # root_path = Path(root)
        # for split in self.__splits:
        #     data_files = list((root_path / split).glob('*.csv'))
        #     for file_path in tqdm(data_files, desc=f'::LOADING {split.upper()} DATA::'):
        #         df = pd.read_csv(file_path, delimiter=';')
        #
        #         value = df.loc[:, self.__feature_columns].values.transpose()
        #         timestamp = df['datetime'].values
        #         anomaly = df['anomaly'].values.astype(int)
        #         change_point = df['changepoint'].values.astype(int)
        #         label = np.logical_or(anomaly, change_point).astype(int)
        #
        #         self.data.append(value)
        #         self.labels.append(label)
        #
        # self.sep_indicators = np.cumsum([item.shape[-1] for item in self.data])
        # self.data = np.concatenate(self.data, axis=-1)
        # self.labels = np.concatenate(self.labels)

    def __check_integrity(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

        for category, category_dict in self.__file_list.items():
            for file_name, file_md5 in category_dict.items():
                if not check_existence(root / category / file_name, file_md5):
                    return False

        return True
