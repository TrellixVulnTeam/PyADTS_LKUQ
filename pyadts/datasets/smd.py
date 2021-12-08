"""
@Time    : 2021/10/18 17:59
@File    : smd.py
@Software: PyCharm
@Desc    : 
"""
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tqdm.std import tqdm

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import check_existence


class SMDDataset(TimeSeriesDataset):
    __label_folder = 'test_label'
    __train_list = {
        'machine-1-1.txt': 'aa0c83b521a709279225c9425d77aca5',
        'machine-1-2.txt': '4838b43da41a01a00df2d9c1b5b9ca8d',
        'machine-1-3.txt': 'e1431bd71306afd202dc8c7a355633ee',
        'machine-1-4.txt': 'e96fe53f95c5407819199b93c70f0602',
        'machine-1-5.txt': '2c7083523e520f2a06cc81ac838d54df',
        'machine-1-6.txt': 'b2d1ec9c0406a37fbe5ee1158ed2d69c',
        'machine-1-7.txt': 'd088355b84247a3e85753f1a58827feb',
        'machine-1-8.txt': 'b15e6ef80411c1f5aaf6cc4be9b123ab',
        'machine-2-1.txt': 'c1e2371361f290ba24f53a4665a0d240',
        'machine-2-2.txt': 'b983092e126a0fd8f4f098c8f2956673',
        'machine-2-3.txt': '2ff04c9f78fb5ee5e551cf25cfebfce5',
        'machine-2-4.txt': '6289cd66f160fbf76c5a79b8119dc825',
        'machine-2-5.txt': '42243978a249565c3e5d4ac027f6e1b7',
        'machine-2-6.txt': 'a4f70d5af404d851a7f1527aaf166ecd',
        'machine-2-7.txt': '321bf8f4219d90740e5e1a905e7a0e23',
        'machine-2-8.txt': '9c38f1c8a422f634c0aad080aa1d5765',
        'machine-2-9.txt': '8a27fdf0d965e8087f6a705db016c153',
        'machine-3-10.txt': '99c95e65c9cc2546d0ffb64ff35646ba',
        'machine-3-11.txt': 'd611855a652b583f34a2a45b35fe6b2e',
        'machine-3-1.txt': 'a1156ca715a12b693cccde67ecf28d29',
        'machine-3-2.txt': '1197c6d7c1d703c8bc04837ae8c5f710',
        'machine-3-3.txt': '583bd37c21929044db3f9c38a5d4ffb9',
        'machine-3-4.txt': '19cfa4e05f80411f8b430fd13cc0e11e',
        'machine-3-5.txt': 'b735d2860ec6a4913ed7abbff6cba9a6',
        'machine-3-6.txt': 'd494e20a311c39a9e398838a0aaf4f18',
        'machine-3-7.txt': '51866b5c59f777ea820df6dc37be9099',
        'machine-3-8.txt': 'b3a147221a18ceb06e81fe433e1b61bc',
        'machine-3-9.txt': 'cb01f1e9abc55139058fd4a4cd100d72'
    }
    __test_list = {
        'machine-1-1.txt': '4f24368801764c42f25dc96f7e0dbc3b',
        'machine-1-2.txt': 'f2ae815e29468ce5be2e9b754accc472',
        'machine-1-3.txt': '7dfb3631214ed7c22fdf8cdedfbe3624',
        'machine-1-4.txt': 'a0575da12d3102f6adfea90b870499e8',
        'machine-1-5.txt': '79314b65341ff51fe0919f8b8b19e02b',
        'machine-1-6.txt': '6e0f08cf5d8ce7441188efd8b9f51cf3',
        'machine-1-7.txt': '74e08e77f7c495f888117e24550d02a3',
        'machine-1-8.txt': '61545bd78e1e72a4593eda0c29dec67b',
        'machine-2-1.txt': 'ee8f02c312ff19c3bd6e2f2f99103345',
        'machine-2-2.txt': '57e3a980d02a8cc29df3241bf316d0ee',
        'machine-2-3.txt': '63065e75beeb6915fd16512aa965e0b0',
        'machine-2-4.txt': '1766d9e525ee0aede60d3eecaac64208',
        'machine-2-5.txt': '9e315c314dd204f537e51c1053a87788',
        'machine-2-6.txt': '2c67f330b154ce90114e1d568d066196',
        'machine-2-7.txt': '4960a62d869fe0cbd98b2e861fbf7ee2',
        'machine-2-8.txt': 'd3341df383600b7645f816e56f12ddcb',
        'machine-2-9.txt': 'e17c175a73649f15ea874f391c8d626c',
        'machine-3-10.txt': 'd9147cf567e27c7353e316b23268fe84',
        'machine-3-11.txt': 'a4b3033cf94d557f8d0bd8f84c471318',
        'machine-3-1.txt': 'fae624306f4977dceb8a022347c1122f',
        'machine-3-2.txt': '2896fc0317fa463b2289a2d36c04466e',
        'machine-3-3.txt': '212eaefd5687b0a09785bf9fa0758196',
        'machine-3-4.txt': '75625043dbcac987e16950ba54398c40',
        'machine-3-5.txt': '1b46d248c078060a1e468cc65bf3d8a2',
        'machine-3-6.txt': '829a1e9946987ad06e32941e3e8a6a84',
        'machine-3-7.txt': '74cad6eff69128a961676ac6bef31d65',
        'machine-3-8.txt': '096bfee232557b650541e9d7ea91948a',
        'machine-3-9.txt': 'cdddb9df219761e7a216ed88f1e61fac'
    }
    __test_label_list = {
        'machine-1-1.txt': 'd67776f7bcaab750b7a11b8eae1fabbb',
        'machine-1-2.txt': 'f38824bb2bd97564a537bf4fc5b7d531',
        'machine-1-3.txt': 'd0225b07258e01ea940c974aa6e05b01',
        'machine-1-4.txt': 'dea2d5cc33923b2be0bc48aa279e5f72',
        'machine-1-5.txt': '429f2c4236f55b3ba8cb216ad7e10f0d',
        'machine-1-6.txt': '958cffa52dc80b1972f91ff68eb9c2a3',
        'machine-1-7.txt': 'f8bdf369191cd2fb3f8401738eaa42d9',
        'machine-1-8.txt': 'b23375d2af3a9f25a3f5f6678ce6ff85',
        'machine-2-1.txt': '078075e7b14a2105446d224c45bd42c8',
        'machine-2-2.txt': 'ba1653b9801e86e1f08f249730ae116f',
        'machine-2-3.txt': '8d989091ef862a90d5b50f8908a29eb8',
        'machine-2-4.txt': '3723479240ce907cdce68c8637e97c4e',
        'machine-2-5.txt': '92b91a30f2947747228dff4ee764652f',
        'machine-2-6.txt': '22709556b66362c931fc8e6fb464c784',
        'machine-2-7.txt': '9d5f3adc7ca92feebd3775ef4e3708e9',
        'machine-2-8.txt': '3b6a1147bc7685de8469a7517d4994da',
        'machine-2-9.txt': '688bb7bf6b5d2386d260a091bf759947',
        'machine-3-10.txt': '6ee630cc805f1c61eb0063c7668b8dfb',
        'machine-3-11.txt': '9d8b590d655dd02a5018cb59abad1da3',
        'machine-3-1.txt': '60375cf7294443b2d117853526621ce5',
        'machine-3-2.txt': '04e78531b1d366771312f6bc5e690da8',
        'machine-3-3.txt': '4fa0845473c4e0c26097cd5975818e83',
        'machine-3-4.txt': 'b68557d0ccc3669d33cac6cbc98118a5',
        'machine-3-5.txt': '00b9158f3b16a25fe6b1162ecabf9101',
        'machine-3-6.txt': 'e21da3b1e8e049a3aa0ed3d1ff90249c',
        'machine-3-7.txt': '347a035d248b66cf490476ec51c8e358',
        'machine-3-8.txt': '05e85de918df349ea70c36cb840d68cc',
        'machine-3-9.txt': '1de3dfa8f7967bc91fcb3fd862316104',
    }

    def __init__(self, root: str = None, train: bool = True, download: bool = False):

        if root is None:
            root_path = Path.home() / 'smd'
            warnings.warn(
                f'The `root` path of the dataset is not set, using user home dir {str(root_path)} as default.')
        else:
            root_path = Path(root)

        if download:
            raise ValueError(f'The SMD dataset should be downloaded manually. '
                             f'Please download the dataset at `https://github.com/NetManAIOps/OmniAnomaly`!')
        else:
            assert self.__check_integrity(root)

        data = []
        labels = []

        if train:
            warnings.warn(
                'This dataset contains no labels for the training set. Thus all data points will be considered as normal by default!')
            split = 'train'
        else:
            split = 'test'

        data_files = list((root_path / split).glob('*.txt'))
        for file_path in tqdm(data_files, desc=f'::LOADING {split.upper()} DATA::'):
            df = pd.read_csv(file_path, delimiter=',', header=None)

            value = df.values
            if split == 'test':
                label = np.loadtxt((root_path / self.__label_folder / file_path.name).as_posix(), dtype=int)
            else:
                label = np.zeros(len(value), dtype=int)

            data.append(value)
            labels.append(label)

        super(SMDDataset, self).__init__(data_list=data, label_list=labels)

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
