"""
@Time    : 2021/10/18 17:59
@File    : smap.py
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


class SMAPDataset(TimeSeriesDataset):
    __link = 'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'
    __label_link = 'https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv'
    __filename = 'data.zip'
    __label_filename = 'labeled_anomalies.csv'
    __splits = ['train', 'test']
    __label_folder = 'test_label'
    __train_list = {
        'P-1.npy': '995c17cfde44a7ff8e15abd959b1e5b3',
        'S-1.npy': 'b5529541c6e127dd741ef47391e4e46f',
        'E-1.npy': 'e0dc4b67d3f218dd91fc37b4aa0f7220',
        'E-2.npy': '5fe47fd3e377f80c2e30589621589a4f',
        'E-3.npy': '956f9ee5d264071ffe07aa26a96b1938',
        'E-4.npy': '864888cdd35c5539fb3072d4c50c6d6a',
        'E-5.npy': 'be3f666294334d5ebf9bfbd3809c1cc5',
        'E-6.npy': 'e149c8ad34888761578bd65133fd7747',
        'E-7.npy': 'aaf589a3ffeec2770f95ec337d7401e4',
        'E-8.npy': 'db80b6dcd31bd27f72580726345fd496',
        'E-9.npy': 'c9e48f5942f2991feb4614ab985967a9',
        'E-10.npy': 'b7f12cf512ed689ce00ea63616046e77',
        'E-11.npy': '714f2853ba8d556707452cd1e4e78dd1',
        'E-12.npy': 'd2b9cef0381f5b9ef755946bfc359a3a',
        'E-13.npy': '362d578f48ef886f78de95ce9c9bfda2',
        'A-1.npy': '62610c0185a465169ebac4e847c6107b',
        'D-1.npy': '29989438c073f0ec3edbe1541e8d7548',
        'P-2.npy': '3acebf335ba3b13778a6011900c7071c',
        'P-3.npy': 'd8ca4a55a374856db5b9ccf847572d94',
        'D-2.npy': '1697812cca7b9f00e53ead03899d9f24',
        'D-3.npy': '6434b7a83397b6be02f754507025cb71',
        'D-4.npy': '6348a22132d49b36eb8b6fd0d16ad6dd',
        'A-2.npy': '857eed41c8638eee47515e5d47862031',
        'A-3.npy': 'd6395e792a83a4f793aa4e40ae996599',
        'A-4.npy': '2fd9bb513d94c8c0f59570685f38c5bd',
        'G-1.npy': 'd72b356e5694b6e4bbe7f24d79b973a6',
        'G-2.npy': '1b52408657eb120fd4d536222e39a62b',
        'D-5.npy': '7a84f32405900b68a3b57ba368e13e07',
        'D-6.npy': '7476fa7ec60d48747c6924b6cd32a6cf',
        'D-7.npy': '369f86c1967d505c4371114495f96f5a',
        'F-1.npy': '2fc56e643705739c5f5c9df72a8062c1',
        'P-4.npy': '37cd6ac0d9e1463cf3866aa9814de99c',
        'G-3.npy': 'a45f17b5d9ffdb79f39d6750db4ae84c',
        'T-1.npy': 'dcf79bcd4668aed5e15ce0e53f4f8c07',
        'T-2.npy': '84fadc0c07739140866d950cd34ea6c3',
        'D-8.npy': '11e45a5966439425832c480b60360fc6',
        'D-9.npy': '71da25798304b4ed75e8c0a3e39a3360',
        'F-2.npy': 'b1d03299ab578335c9c35c62162bdd00',
        'G-4.npy': '05e8b4c446a42422fec035f590273db3',
        'T-3.npy': '30460259d5176e04d676b3321252667f',
        'D-11.npy': '577d23f96997f503e60f14817d2067ed',
        'D-12.npy': 'c9b0ce0bedc4f2fd6ee0563373d5527d',
        'B-1.npy': '4836aa89b278bb7ac054c236db59b5ae',
        'G-6.npy': 'b796266cdb858e6a7f4b88ea901dc749',
        'G-7.npy': '809a8a7effb1a1edf3b46fd7676cc342',
        'P-7.npy': '41435dabdcdc8e275d90dfed10034361',
        'R-1.npy': '6047d8f7b498b751fff8ce4835e50bd1',
        'A-5.npy': '835e5991d8b07b057f9876dbef7b9704',
        'A-6.npy': '05b76c9e8ecc36a881165b3f63f85645',
        'A-7.npy': 'fbcc186f6e13936c610d5bd9f1270dcd',
        'D-13.npy': 'ecf21d8436351ef80a40ec49c29e585a',
        'A-8.npy': 'ee3a0b6f400d856d89a349f25bde1194',
        'A-9.npy': 'b330c4ed5ebabc120ba465076d0a6957',
        'F-3.npy': '28990f8f9694ff112ed8e71ff71f95bd'
    }
    __test_list = {
        'P-1.npy': '610579e3e10633fac9da11be10c8abfa',
        'S-1.npy': 'e9d5ae1cf1fa2bf61c6ecee0f554215d',
        'E-1.npy': '22404262a4afd134ae41feca5d35deba',
        'E-2.npy': '00094718b432157049f52ce303ab5b2a',
        'E-3.npy': '9bbb43df1bd44e03dc42329ee2c8a19a',
        'E-4.npy': '7358bd4d6756a8c631ee738cedfda07d',
        'E-5.npy': 'cafa10e324757da72cc0470f451b3175',
        'E-6.npy': '157ef1689976383ab5e48ca7c2f859ec',
        'E-7.npy': '0e2313a32f7b08bc2ef8be8da5bd5190',
        'E-8.npy': 'cfb40cf755b42246b770fa5000d3135e',
        'E-9.npy': '30374820df1a542b28cb5cd36d7178f8',
        'E-10.npy': '3a00e77c78f03920363ec7bb1e43c745',
        'E-11.npy': '6b30b5c72e451f1a0de1f0c588e8c6e5',
        'E-12.npy': '2011b83118303d25c48d66a8e1677517',
        'E-13.npy': '6f5ca26e2d8efdd7b8311ab516e97751',
        'A-1.npy': '2b98a4482850e912c987d3d6eabdd94c',
        'D-1.npy': 'e678b8fd0f9926a5806b51fe309d2a6a',
        'P-2.npy': '5023f9d53ad24651affc0fc660b6c70b',
        'P-3.npy': 'a3dc791a5846f2ca246b4ff39f090eb7',
        'D-2.npy': '2c2ca58bca07dbd892a03889a69ce493',
        'D-3.npy': 'ff17ff9ef7d5d12737e454379a2f5a50',
        'D-4.npy': 'ae89fbb4be5d3a7cd54f71826bb725ad',
        'A-2.npy': 'c2482d64d7ec913a585109a8e35d8e31',
        'A-3.npy': '081d23e35e1010f51b113100ba9c2272',
        'A-4.npy': 'f33691e7ba4001b015864fe8164ba764',
        'G-1.npy': 'aceeca310763eec509e6766c163ee2ff',
        'G-2.npy': 'ef8afb62b7e0a16aa4523798c33123ed',
        'D-5.npy': '0919298af55b8ab238a1956978d6db9e',
        'D-6.npy': 'c3aa85a84c9a5be98af4f4d4dd0ad13c',
        'D-7.npy': '0b3641708b09c88cfd8a9f631e0fbe0d',
        'F-1.npy': 'f3a60f4337b34dedde55a08422d6694d',
        'P-4.npy': '894476d54ae05001d1b04f6c80d99208',
        'G-3.npy': '1a8b3db55ab59d5e6514a96416fba45e',
        'T-1.npy': '72b12f21b197aebd4c09cdce3f99f702',
        'T-2.npy': '1c2a522b2c56a886e55a202c4a594d25',
        'D-8.npy': '5663e43d24db9b3dbbc92aedd2dc9c63',
        'D-9.npy': 'ec6487c7bbb86eada3d627c422780b79',
        'F-2.npy': '9296d99cea3d788a973d1e86e7d003a8',
        'G-4.npy': '8191db3dc17d9f87472f24ae32d0ff06',
        'T-3.npy': 'b058b28590bb5a98748a94766afe08e0',
        'D-11.npy': 'f065d9d27e360ebee61e16eefcbb00c3',
        'D-12.npy': '1b51bc0d491b3c641869d9c9ab4208ca',
        'B-1.npy': 'b969001f9b1852e8ab7ddc5c8d6561be',
        'G-6.npy': '24f3df1b4449b6c597bd109985925cd7',
        'G-7.npy': '41be65c0de1f610f1f70aff672bec244',
        'P-7.npy': '207cb64eb0ba6265948f04745fafc668',
        'R-1.npy': '097d5faf0fb7d8e4722ecd743ff179b5',
        'A-5.npy': '175606cac8de18be18e607ec68c29dac',
        'A-6.npy': '9c729d6b7b659910a17441a76e538b81',
        'A-7.npy': 'a8a8dc9043a9d3822dff2644b15c9a71',
        'D-13.npy': '3512572e8f22192b0e3e6efb5d8da8d8',
        'A-8.npy': '09931c699b6d4054d893e20be67d0532',
        'A-9.npy': 'fea96c2f7643f653f122610d19b78839',
        'F-3.npy': 'c51d23deaf31e0c5c8a6db0d8e84b5e2'
    }
    __label_md5 = 'c54f6c09f44410763e33aac7329d769a'

    def __init__(self, root: str = None, train: bool = True, download: bool = False):

        if root is None:
            root_path = Path.home() / 'smap'
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

        super(SMAPDataset, self).__init__(data=data, labels=labels)

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
