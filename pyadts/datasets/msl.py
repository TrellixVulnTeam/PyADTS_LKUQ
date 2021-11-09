"""
@Time    : 2021/10/18 17:59
@File    : msl.py
@Software: PyCharm
@Desc    : 
"""
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
        'C-1.csv': 'ad99c084f78a137f816975fec4e0aa79',
        'C-2.csv': '9a056efa22df6ba3ac3a3b41b1bdd1e8',
        'D-14.csv': 'ae4e9253ad14ab145c1e97b952627b74',
        'D-15.csv': '3692ed6156278cc46d995fc80058ad5c',
        'D-16.csv': 'efd36a140179a9a78299478d0a0e0a77',
        'F-4.csv': 'd5e30703811187b289474a7d2b53dea7',
        'F-5.csv': 'b5db27d9b0bdc63aa1091f5af88ae098',
        'F-7.csv': 'ea74e9b0be94910bc42507e999bfc628',
        'F-8.csv': 'cf5942b1d1ee9da7d35f0b8df0c7f7fc',
        'M-1.csv': '1d1a609d2ef298f78994b67d29ca6244',
        'M-2.csv': 'f4c0f047f088cc59516cec6f4ffb8404',
        'M-3.csv': '63d0e40c4534654072e5d0447bd04d3e',
        'M-4.csv': 'a4e80120c82e4769e993593c6dafc541',
        'M-5.csv': '6787f132da99c30854e48ae878c0b3f8',
        'M-6.csv': '9f796e5d817d96e6ee0249c58bddd5af',
        'M-7.csv': '2b2b2aaeb1f964cc337caf2a98d38c97',
        'P-10.csv': 'd53bbd361bb95f55c55ea5508e96ba16',
        'P-11.csv': 'd889b35d559db6e67bdb29461ce7fb64',
        'P-14.csv': 'b7a9c7d558858da37497e91ab7a53966',
        'P-15.csv': 'd2673010e5ca552b6c00dca7de673536',
        'S-2.csv': '4b257042d3a67ca01a844d1de82826f9',
        'T-12.csv': 'a7eb848479e519456bce4273de9a62cb',
        'T-13.csv': 'cbcf2997b3049bcab8d99d8a8654d9bd',
        'T-4.csv': 'b7e89d92e305a916609c4cb7eecef2d5',
        'T-5.csv': 'd69453471f931df355c82123116fb1dc',
        'T-8.csv': '3b7ce3b7ccecfcc251bba0d3c21fdc8f',
        'T-9.csv': 'abd0967ca3140b4eb8b0de136dba4dd6'
    }
    __test_list = {
        'C-1.csv': 'b2e6c9dc2e332528b58e7a0f286c7a7a',
        'C-2.csv': 'b4befa2faca2024d74c2012760a9be9f',
        'D-14.csv': 'bac89c507291aebef3df5de780b36823',
        'D-15.csv': '76f20e4a2863a1122ab35f13b6754a75',
        'D-16.csv': '28f809952d467d0429d9cfbf5094d5b5',
        'F-4.csv': '84b95658bae993f1eb3325001a025f03',
        'F-5.csv': 'ef7267c248db9df0457f1aed026e481d',
        'F-7.csv': '95da351a06fb1171ba0c360c74e5826f',
        'F-8.csv': '5f11e83fa36d9f8afa17fa3fddc92ec8',
        'M-1.csv': '82eb43c4c17f8f4791a1d2e64b1b807a',
        'M-2.csv': '77e4c73f803ba423caaf2457cc79f0f6',
        'M-3.csv': 'e1c70b41d4dbd47fd0bce55c8ef403f9',
        'M-4.csv': '03a1fd123f7038a339904e9afed7e083',
        'M-5.csv': '2cdc2c0d2d1292cbd9e0f88938b5c108',
        'M-6.csv': '8edbf28f7b53da25585036521264b957',
        'M-7.csv': '40827ee1576e7ed6aa19dd8a7ab84b6c',
        'P-10.csv': 'cf97d70471dcb751ee27b487fa7ecbe7',
        'P-11.csv': 'a8e1e0997a4df8151fc7ef38684c6753',
        'P-14.csv': 'cf97d70471dcb751ee27b487fa7ecbe7',
        'P-15.csv': '8164fda48cddc7017270cee2bf59bd7c',
        'S-2.csv': '0fd0b8ef86744ffc705ba92c0f1d77d3',
        'T-12.csv': '96578b699c847535cb588736df9baa26',
        'T-13.csv': 'a6c561c58d05c499da6c7bdd7cc6f20a',
        'T-4.csv': 'cb24c226a4cdf531562fcc7e045518c5',
        'T-5.csv': '087c75d039e63946ec45481e746b7f43',
        'T-8.csv': 'cd3f4e5ce118201c25d5596c83d8f5be',
        'T-9.csv': 'a69ba0afa160cd0e9be2f10c9d42dfa3',
    }
    __test_label_list = {
        'C-1.csv': 'a6b3f44b51abbde0eda64e4b55fec2ae',
        'C-2.csv': '108cd4221af4fe223cc55baff76a9c84',
        'D-14.csv': '2f650c35b1611911b58923238491e432',
        'D-15.csv': 'e0a483e47aaa0f8ad711f37b69fc9c1c',
        'D-16.csv': 'df733e28712587ad855d259ee31b9f36',
        'F-4.csv': 'bf5bedae6b7fcde81c086e733da6c2cb',
        'F-5.csv': 'adaf85e8fcdc5d0dbbe099f82854c0cc',
        'F-7.csv': '8ba94d6e0f58eeb145c81e1e34b8e9c6',
        'F-8.csv': '4331d328da82c1aee7b716b96ac4934d',
        'M-1.csv': 'b91ec9c16b64335e86b0c8da834b77c4',
        'M-2.csv': 'b91ec9c16b64335e86b0c8da834b77c4',
        'M-3.csv': '5e7d5a4364a3f82b13dcb7ac5c0d16af',
        'M-4.csv': '760d2df993dfc4133ea7429bd683f0fb',
        'M-5.csv': 'dddff76339ad65ebf913dff55767d562',
        'M-6.csv': 'f36529732f2fb60fa08f9a6bcb5cdead',
        'M-7.csv': '51d4654c19d4cf2b539341a3f1b9f668',
        'P-10.csv': 'f067c04ee03497b50fc5891f1f8eb076',
        'P-11.csv': '3e34043c80af6e0617b8b3c6a2643f0c',
        'P-14.csv': '7d31ca67b01f972eee50bb10e3dff573',
        'P-15.csv': '4187ba2cf2c87f7a13791b36e88289ca',
        'S-2.csv': '81186b223602fbc800ad8dbe02e39b54',
        'T-12.csv': '7b7ca21d65a6b6d72ad5431a2d968d08',
        'T-13.csv': 'e986de800f0c066aa158d5306c4ca711',
        'T-4.csv': '6a5260b002df32a7bcfb7bc28d43d6b2',
        'T-5.csv': '43d9b5c8502d30f92daefa7b993bc1cf',
        'T-8.csv': '782c15d88e9530d972063b9391bc01bc',
        'T-9.csv': 'd5227f2d2b79330f16944401b6144956'
    }

    def __init__(self, root: str, download: bool = False):
        super(MSLDataset, self).__init__()
        root_path = Path(root)

        if download:
            if self.__check_integrity(root_path):
                print('Files are already existed and verified.')
            else:
                print('Start downloading...')
                download_link(self.__link, root_path / self.__filename)
                decompress_file(root_path / self.__filename, root_path)
        else:
            assert self.__check_integrity(root)

        self.data = []
        self.labels = []

        root_path = Path(root)
        for split in self.__splits:
            data_files = list((root_path / split).glob('*.csv'))
            for file_path in tqdm(data_files, desc=f'::LOADING {split.upper()} DATA::'):
                df = pd.read_csv(file_path)

                value = df.values.transpose()
                if split == 'test':
                    label = pd.read_csv(root_path / self.__label_folder / file_path.name).values[:, -1].astype(int)
                else:
                    label = np.zeros(value.shape[-1], dtype=int)

                self.data.append(value)
                self.labels.append(label)

        self.sep_indicators = np.cumsum([item.shape[-1] for item in self.data])
        self.data = np.concatenate(self.data, axis=-1)
        self.labels = np.concatenate(self.labels)

    def __process_data(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

    def __check_integrity(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

        for key, value in self.__train_list.items():
            if not check_existence(root / 'train' / key, value):
                return False

        for key, value in self.__test_list.items():
            if not check_existence(root / 'test' / key, value):
                return False

        for key, value in self.__test_label_list.items():
            if not check_existence(root / 'test_label' / key, value):
                return False

        return True
