"""
@Time    : 2021/10/18 17:59
@File    : msl.py
@Software: PyCharm
@Desc    : 
"""
import os
import shutil
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tqdm.std import tqdm

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import check_existence, download_link, decompress_file


class MSLDataset(TimeSeriesDataset):
    __link = 'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'
    __label_link = 'https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv'
    __filename = 'data.zip'
    __label_filename = 'labeled_anomalies.csv'
    __splits = ['train', 'test']
    __label_folder = 'test_label'
    __train_list = {
        'M-6.npy': 'ed843065ceff277a1457f9e90082130c',
        'M-1.npy': 'df16e81ff409cbadd39271127b63a64b',
        'M-2.npy': '28a20778dc22612f1485d726a383e0e3',
        'S-2.npy': '2a39fe1a4c0689eb7739a7e1998ca5e0',
        'P-10.npy': 'df56794684109c03c97f56c039f2f0af',
        'T-4.npy': '94248ba137006f14dccb300408f3e2a2',
        'T-5.npy': 'cd1c4c0dbc55b4d76211229c6d3c7651',
        'F-7.npy': '25c488b38d4721d229df9de452561da8',
        'M-3.npy': '327314369ae1e172f7bca78577954993',
        'M-4.npy': 'cccbe9fd9c0549226db403793e4938fc',
        'M-5.npy': '2df9a1793bae720a856fbbc268b3c161',
        'P-15.npy': '5eb87b6979217310a9c1a848134330ca',
        'C-1.npy': '1f0314698d6bbd21e662f89962886585',
        'C-2.npy': '18d6ea54f2dd0fbb7fb683330f947dd2',
        'T-12.npy': 'cf9981b61766d7d90d3344a41ae74fa3',
        'T-13.npy': '3c7dbce7fa6c1731f2a151f5623c140e',
        'F-4.npy': '7570a461379e220657a912c0d43a05c9',
        'F-5.npy': 'ff907a54d4a62c5adbd952bf55cbf813',
        'D-14.npy': '1cd8fc2df339e6a88c421fcd82645342',
        'T-9.npy': 'f6cad7f14fcc3f2f218645120421b197',
        'P-14.npy': '0fb447698e6fdb5b73200f06ebafb344',
        'T-8.npy': '88c5687a15f3fe6cc8221f263c8ac921',
        'P-11.npy': '303b2bb862bea8dddffd09a24d40f9f5',
        'D-15.npy': 'a854b798e192547024e1667586b7f19c',
        'D-16.npy': '7dcef310d3e393a203491a24d2350930',
        'M-7.npy': 'fa5262d17aec9cbd49381ef488ae96eb',
        'F-8.npy': 'ac4cbca12bfb5fbcfc61c06c4ee834c5'
    }
    __test_list = {
        'M-6.npy': '0ab9799029e790071b73ae67ca3cd82c',
        'M-1.npy': 'e6e269bd6830a2109d7c2862a8b1362b',
        'M-2.npy': '84474717eaa3882c548448b62547f561',
        'S-2.npy': '9793b166fd7dd850aef0564b41e8c58b',
        'P-10.npy': '6177bb0c469fb8d055c6f9f852badae4',
        'T-4.npy': '77476e836756e141a854d97f11e1ea7c',
        'T-5.npy': '1bb9238629251f34400213e215e3ca81',
        'F-7.npy': 'cdf91dc2c24e065c4d919b0af34cb97a',
        'M-3.npy': '0ac6f7e3093b24167a435691da18c040',
        'M-4.npy': '9286240bb679728d092a3e803e28f1d8',
        'M-5.npy': '9f549a24bbb551f28656cef17af3aa4c',
        'P-15.npy': '505fcd65ff89e137610c8883f92515ac',
        'C-1.npy': '02400e614a7ac3388af5985b18770993',
        'C-2.npy': 'cb7a060ef0649e553a96ab38fdd14aa6',
        'T-12.npy': '4836e17279bf4a07b930687ec6fd7def',
        'T-13.npy': 'c47c4f80d7a7fc1e85cfe3b668024297',
        'F-4.npy': '2f698f24e6278d881f7ebfe1d32abdc2',
        'F-5.npy': 'a1a11e7afebf82ed0fd8e45e07d64152',
        'D-14.npy': 'ea9726366943e40285a5e7c633b5aedc',
        'T-9.npy': '462008901e4e6f310cebbe3cc1aff31b',
        'P-14.npy': '6177bb0c469fb8d055c6f9f852badae4',
        'T-8.npy': '94fa78f46f1e14abf483b4e87aa9c159',
        'P-11.npy': '1a636597948732d79d64f363448dc083',
        'D-15.npy': '6ffc7b44355bbf712c9ad13bdeaf9beb',
        'D-16.npy': 'd553c2b10c144a0a1fae540a4e2d5f26',
        'M-7.npy': '4cc829c882f09cb7c7066fc44775be3b',
        'F-8.npy': 'e47323315a8ac0d01dc5b06ca06eec31'
    }
    __label_md5 = 'c54f6c09f44410763e33aac7329d769a'

    def __init__(self, root: str = None, train: bool = True, download: bool = False):

        if root is None:
            root_path = Path.home() / 'msl'
            warnings.warn(
                f'The `root` path of the dataset is not set, using user home dir {str(root_path)} as default.')
        else:
            root_path = Path(root)

        if download:
            if self.__check_integrity(root_path):
                print('Files are already existed and verified.')
            else:
                if not check_existence(root_path / self.__filename):
                    print('Start downloading...')
                    download_link(self.__link, root_path / self.__filename)
                if not check_existence(root_path / self.__label_filename):
                    download_link(self.__label_link, root_path / self.__label_filename)
                print('Start decompressing...')
                decompress_file(root_path / self.__filename, root_path)
                os.makedirs(str(root_path / 'train'), exist_ok=True)
                os.makedirs(str(root_path / 'test'), exist_ok=True)
                for data_file in self.__train_list:
                    shutil.move(str(root_path / 'data' / 'train' / data_file), str(root_path / 'train'))
                for data_file in self.__test_list:
                    shutil.move(str(root_path / 'data' / 'test' / data_file), str(root_path / 'test'))
                shutil.rmtree(str(root_path / 'data'))
                assert self.__check_integrity(root_path)
        else:
            assert self.__check_integrity(root_path)

        data = []
        labels = []

        label_df = pd.read_csv(root_path / self.__label_filename)

        if train:
            warnings.warn(
                'This dataset contains no labels for the training set. Thus all data points will be considered as normal by default!')
            for data_file in tqdm(self.__train_list, desc='::LOADING DATA::', colour='cyan'):
                data_item = np.load(root_path / 'train' / data_file)
                data.append(data_item)
                labels.append(np.zeros(data_item.shape[0]))
        else:
            for data_file in tqdm(self.__test_list, desc='::LOADING DATA::', colour='cyan'):
                anomaly_sequences = eval(
                    label_df[label_df['chan_id'] == data_file.split('.')[0]]['anomaly_sequences'].values[0])
                data_item = np.load(root_path / 'test' / data_file)
                label = np.zeros(len(data_item))
                for seq in anomaly_sequences:
                    label[seq[0]: seq[1] + 1] = 1
                data.append(data_item)
                labels.append(label)

        super(MSLDataset, self).__init__(data=data, labels=labels)

    def __check_integrity(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

        for key, value in self.__train_list.items():
            if not check_existence(root / 'train' / key, value):
                return False

        for key, value in self.__test_list.items():
            if not check_existence(root / 'test' / key, value):
                return False

        if not check_existence(root / self.__label_filename, self.__label_md5):
            return False

        return True
