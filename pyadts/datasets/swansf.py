"""
@Time    : 2021/10/29 0:05
@File    : swansf.py
@Software: PyCharm
@Desc    : 
"""
import shutil
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tqdm.std import tqdm

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import download_link, check_existence, decompress_file


class SWANSFDataset(TimeSeriesDataset):
    __link = 'https://bitbucket.org/gsudmlab/mvtsdata_toolkit/downloads/petdataset_01.zip'
    __filename = 'petdataset_01.zip'

    def __init__(self, root: str = None, download: bool = False):

        if root is None:
            root_path = Path.home() / 'swansf'
            warnings.warn(
                f'The `root` path of the dataset is not set, using user home dir {str(root_path)} as default.')
        else:
            root_path = Path(root)

        if download:
            if self.__check_integrity(root_path):
                print('Files are already downloaded and verified.')
            else:
                if not check_existence(root_path / self.__filename):
                    print('Start downloading...')
                    download_link(self.__link, root_path / self.__filename)
                print('Start decompressing...')
                decompress_file(root_path / self.__filename, root_path)
                for csv_file in (root_path / 'petdataset_01').glob('*.csv'):
                    shutil.move(csv_file, str(root_path))
                self.__check_integrity(root_path)
        else:
            self.__check_integrity(root_path)

        df_list = []
        for csv_file in tqdm(root_path.glob('*.csv'), desc='::LOADING DATA::', colour='cyan'):
            df = pd.read_csv(csv_file, delimiter='\t')
            df.sort_values(by='Timestamp', inplace=True)
            name = csv_file.name
            label = name[name.find('[') + 1: name.find(']')]
            df['label'] = label
            df_list.append(df)

        df = pd.concat(df_list, axis=0, ignore_index=True)
        df.sort_values(by='Timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['label'] = df['label'].map({'NF': 0, 'C': 1, 'B': 1, 'M': 1, 'X': 1})
        df['IS_TMFI'] = df['IS_TMFI'].map({True: 1, False: 0})

        # TODO: filling NaN and None

        labels = df['label'].values
        timestamps = df['Timestamp'].values
        df.drop(columns=[col for col in df.columns if ('loc' in col or 'label' in col or 'Timestamp' in col)],
                inplace=True)
        data = df.values.astype(np.float)

        super(SWANSFDataset, self).__init__(data=data, labels=labels, timestamps=timestamps)

    def __check_integrity(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

        return len(list(root.glob('*.csv'))) == 2000  # for simplicity
