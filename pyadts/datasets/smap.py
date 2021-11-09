"""
@Time    : 2021/10/18 17:59
@File    : smap.py
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


class SMAPDataset(TimeSeriesDataset):
    __link = 'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'
    __label_link = 'https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv'
    __filename = 'data.zip'
    __label_filename = 'labeled_anomalies.csv'
    __splits = ['train', 'test']
    __label_folder = 'test_label'
    __train_list = {
        'A-1.csv': '4e30e2cdbf81a659d4c01e6b42aea341',
        'A-2.csv': '4fefd422262baa944cbb4db25b3bb066',
        'A-3.csv': '957f7cc7dd720fcf8ebef4914143bdba',
        'A-4.csv': '3db3a0b418937ec05b56eee7e641b4ed',
        'A-5.csv': '695c093838507b39a1ee0437549090ec',
        'A-6.csv': 'bcb8374c1a4301e3944e4ebb9fd71fce',
        'A-7.csv': '73271b5b810c91418d07aaf31be995ee',
        'A-8.csv': '593772baa3d1c2b5a5d2535303052216',
        'A-9.csv': '0ed327fe1c2270193edd4572b70240d6',
        'B-1.csv': '8a7474e4e476971902503a07ffc3ab19',
        'D-11.csv': '73c192beb6702ef9791e167d58735b58',
        'D-12.csv': '81ec4776f57054032dbb0766b8823507',
        'D-13.csv': '315d2a1223b76492449a43cbc5f82c41',
        'D-1.csv': '8b7a8480b5c119558e944527fea846f5',
        'D-2.csv': '28ba5eaaf328756e6e15c49afcd2378a',
        'D-3.csv': 'bc18b7d01393eb469b40414f188ce7c5',
        'D-4.csv': 'ab9ef1a45d9030e57743740e5cbdbd09',
        'D-5.csv': 'd4c1032b72080ed7194c417fbb323c80',
        'D-6.csv': '9dc7267f6e7e8c26b2c26d6d35ebb5ed',
        'D-7.csv': '8f5002c0c41448080162861d12ab5878',
        'D-8.csv': '6d446636ac92d8bc081ebce49a8edc05',
        'D-9.csv': 'b686cf7ea0480ecb18bb4ae5419aa4d4',
        'E-10.csv': 'ed3c469a4a350d76aebfde42ec9caf42',
        'E-11.csv': 'a0487a644f8e33324bcae2939f85346a',
        'E-12.csv': 'e37f7fae505d8c031d6a2d1622363906',
        'E-13.csv': '442288546be756f3be4ac286e9426c01',
        'E-1.csv': 'c8f9e56c7d1b0f4b2a99a7dac5e54ba8',
        'E-2.csv': '516bcacb694aa753a2cf7a2b00968a3b',
        'E-3.csv': '25dfac6f691a8a981163fd2baf130574',
        'E-4.csv': 'cfa4779abfa7f298dde5c4279c231cfd',
        'E-5.csv': 'e961680e203705ec3bc180cc8d6cbde1',
        'E-6.csv': '72e91b487614cd474d083c40438ffd8a',
        'E-7.csv': '148c47b90afc3a467ebc7d5ed9d95f78',
        'E-8.csv': '9a924fe18133269cfc761718f11867a9',
        'E-9.csv': '9a5a0157104847ffdf312cb145513bd2',
        'F-1.csv': '3a4036065b81274cb7955a63a0ee25e5',
        'F-2.csv': '97c2352ceaa17a7a1299ca7c3e80fa11',
        'F-3.csv': '344907fb0ee38f7dea35402b2da1c8fe',
        'G-1.csv': '16efe3f7ce58b5650ff1e50c85f11e84',
        'G-2.csv': 'c34e46ac1a9d691a9928dad2615f038f',
        'G-3.csv': 'c19a7e3d8a0625454be30d094d6a8bbd',
        'G-4.csv': 'c3cc236e18e942a70692ccfe957fc4bc',
        'G-6.csv': 'b07794c2285db89424809c8d7fb6ed1a',
        'G-7.csv': '0df703ba988d879cbd3ec5d2cafca2e0',
        'P-1.csv': 'dabfdc4fcc7873437366839cb8e46ba5',
        'P-3.csv': '5f053c4e0c6b7683c392669b106d5777',
        'P-4.csv': '6780f34ce184a6b026cacefe6a15dacd',
        'P-7.csv': 'd3b5586b53ee5aac31427eb659f0eed3',
        'R-1.csv': '92a3b630dc4469125d0f141d7fb3632c',
        'S-1.csv': 'c0afd4b77cd003740994f9a5955fa1ee',
        'T-1.csv': 'b2e392d51f53d12d633d7d33ab44215e',
        'T-2.csv': '41c9bf68d220d69eb3e69ae2723c7891',
        'T-3.csv': '53a3df91341029c4b1c8ae597bd906cc',
    }
    __test_list = {
        'A-1.csv': '1da1ac716cd8cd59f0f4faa32384c049',
        'A-2.csv': '986db90bd85b0e3990d7c8b8038944f3',
        'A-3.csv': '27e3b67b01852347a270e23546465d8b',
        'A-4.csv': 'bc95834336a025759ee7649bbf9c8d04',
        'A-5.csv': 'c7e51a3a14ab130f1c26adf92c0f0787',
        'A-6.csv': 'a59e91736bdacc9ba1a3ca71996ec80a',
        'A-7.csv': 'a7ec4733a9aaf1d2fb5030aa233280d6',
        'A-8.csv': 'f96e34e6084480f52712c8ea750b0af5',
        'A-9.csv': 'ca281bb25082841e6f8465ad41a4b13a',
        'B-1.csv': 'da503290846ad9b31cd045b37bad8326',
        'D-11.csv': '0bc92ff5c9ede0df60e48c9c2d79dd71',
        'D-12.csv': '9918ab28da5c873abc488b77744decd4',
        'D-13.csv': 'f9d33bb1f65caf38c6eb947ae395588e',
        'D-1.csv': 'e5d62c2989fe96df49db6776a9649b93',
        'D-2.csv': '4bfea663efa74d0f425a2919f5663fcf',
        'D-3.csv': '468a51c8856bbca862ff30717b7cf027',
        'D-4.csv': '417510185709caeaeb094b20039f691a',
        'D-5.csv': '9fda1b308c2a16dbfd57686b2431679d',
        'D-6.csv': '313061694e2cfd4559f2127ae16b70f5',
        'D-7.csv': '5e461fbabe4c39cac6a56cc8dc7940b3',
        'D-8.csv': '490e8be75d3cdfa57b1dfbb77cba4cb3',
        'D-9.csv': '4af275078bf65e29631947c0f131793a',
        'E-10.csv': '9a085a111817bd77a10cc449e088400e',
        'E-11.csv': '0c92f5ebcfa627151b29d6ac93bada02',
        'E-12.csv': '838636a57c787430fbc4f974f4a58ba2',
        'E-13.csv': '35ea9e98f3e4ee306db8ce4c571e0c55',
        'E-1.csv': 'd3ad70c4e421c2e03996d37f9ac5a31d',
        'E-2.csv': '7ebc939945920ae6118cb817598a01b7',
        'E-3.csv': 'b3904875e55e2cddaa1768da93a20ab6',
        'E-4.csv': 'a7cab1a1b1b8d65a973b48e558778aa5',
        'E-5.csv': 'f5e744e01095dcf863d9dd2ee30c5d2b',
        'E-6.csv': '16f2873668eaaefb387e443374414ca2',
        'E-7.csv': '0b3dcdca8bbc6486e467f12836c89f6e',
        'E-8.csv': 'a7b8c696cd6a9d9e134ec36578eaae07',
        'E-9.csv': '3a24a2b156d213398caac52b875b71ce',
        'F-1.csv': 'ff52a6f53ccd385cce8ef3562ea87a42',
        'F-2.csv': '56cd55f851db0a6cd682acbe29381259',
        'F-3.csv': 'b797c44c8d77756c2d949a1b9ea66171',
        'G-1.csv': '4f16c9b4bf4fb886fe4ec51cdd21017e',
        'G-2.csv': '5d480e9e7945220aa664c9a50e4d831c',
        'G-3.csv': '0c4e2bf60ac093c6c6c6c92ac707ddf0',
        'G-4.csv': 'c640d7659f93133e08ad0eeb341c1800',
        'G-6.csv': 'd8bad8a6df5f071de6af3ee69554ca8e',
        'G-7.csv': 'df8e7a2dd2a5c478e3c9beea14abcbac',
        'P-1.csv': '81ee610d86f623ed4c45885b5fc0cc46',
        'P-3.csv': 'a136336ca8dacf59878de2b1389cddf4',
        'P-4.csv': '40e3a23260a3c0f684a27ec89b66ed9a',
        'P-7.csv': '16d45a508a94fd25fee5afa1528d4d24',
        'R-1.csv': '1a667593b3065c7bb34d436058e5cdee',
        'S-1.csv': 'febe2b821d19c55eb18e6c7eac08b36b',
        'T-1.csv': '2043e008aab633fa2872689aae78b215',
        'T-2.csv': 'e62f209221c842c87d0feb1e1235de46',
        'T-3.csv': '01204c0d3816bd337965672a125b889f',
    }
    __test_label_list = {
        'A-1.csv': '2c07e48c4effdcee64f039c1578528e2',
        'A-2.csv': 'ff3dd92e1466cc7ccba2efb595b80929',
        'A-3.csv': '831138c9bd03fed43ff4c5bb9a7930eb',
        'A-4.csv': 'e3e694a14fa3ac093099e2455e54fe21',
        'A-5.csv': 'fc8cf26121ae1ab9d0bdf6caf7171ee4',
        'A-6.csv': 'fb98e44c5d15be17e84a330abfb73ddf',
        'A-7.csv': 'f4f89dc6c743bd967d58774974334818',
        'A-8.csv': '88dae0f09a98da8e64ebcaf936ebd9e0',
        'A-9.csv': '798ebac52f9d704e160e628721a5631c',
        'B-1.csv': 'e88a902782cf86478e8f2a7c83d561b5',
        'D-11.csv': 'adcf3b49ef83a276e46961b44ba9509e',
        'D-12.csv': 'c3b0e11e7d75d944446ea88a95cb852d',
        'D-13.csv': 'b4065447f7afdc2923d53bf2bce87b1e',
        'D-1.csv': 'e451ae78eca9e19ddfb8e559c679fa53',
        'D-2.csv': 'e3f780a632d5772fc6687dcde5639f76',
        'D-3.csv': '2a0824e4970138fa852e538b5d9edb16',
        'D-4.csv': 'a8f1d0fc60a5ba304ae1491dd3610d7a',
        'D-5.csv': '7c69b80009428a435e62a8e6a8507424',
        'D-6.csv': '1aeb64f7e4132a07458176a5d56f4229',
        'D-7.csv': '4be3ccb7106c52406e085a0d305bb357',
        'D-8.csv': '1342d95b4ebd02b2ccdabb32816c6802',
        'D-9.csv': 'bbfc116aaf12c84a4c6a1d9d41399ff0',
        'E-10.csv': '92c876f95fad8f8d2565db1ac25fbeac',
        'E-11.csv': 'f08db33a974ecb8e1cc3e55ae8205561',
        'E-12.csv': 'd5efc1db8b506f890d654ce2f1be5fd0',
        'E-13.csv': 'cc50b3e960ba7668990f90f5a5520328',
        'E-1.csv': '603eec8e33b2e0a6cdcd666dd232ae75',
        'E-2.csv': '9e0489322435dc2add62aa5aa1cd30ef',
        'E-3.csv': '93556595e5f0964e324cf7a83feffb4e',
        'E-4.csv': '23dfff7f47c881789d155942c4fc1bbe',
        'E-5.csv': '04a1b486ee8ae26bb3f0efdb8b6563d7',
        'E-6.csv': '0bd960a686145ee299b40e9baa124822',
        'E-7.csv': 'd83f868d6c2632992eb39be2916e1623',
        'E-8.csv': 'e70397d7bf98e7ce2ec9b03a825dff75',
        'E-9.csv': '0abe1235b0868a49dd23361cf96f3f67',
        'F-1.csv': '58ada31e3bd2ecaf8c3d9657f07dc22c',
        'F-2.csv': 'af76241aa9ce11cdaeb533c17a972736',
        'F-3.csv': '327a3b997e9c3b295464b9c3d0060598',
        'G-1.csv': '00deb33f4a9533a4743d3920de36d016',
        'G-2.csv': '4beb2316dbd0ba2cb76fabd9496e8856',
        'G-3.csv': '297e52181afee8ce58e265ce938aeef9',
        'G-4.csv': 'dfcdffcc7a8f5814276d169ed97854d5',
        'G-6.csv': 'a05104d5742f3f521a682b8ef6802aca',
        'G-7.csv': 'e9302bfe902b279b1c310cc19f5e9832',
        'P-1.csv': '7102dca68a8465097172c6496b02d815',
        'P-3.csv': '874cad35106e614b6d5880ba77579d8e',
        'P-4.csv': '7c7fec73a27def06c167216e9cbd3c9a',
        'P-7.csv': 'cc9da0397744f06b38daaf95eeaf4692',
        'R-1.csv': 'b3e1bf55c4f9e272c932e088b91a2b9a',
        'S-1.csv': '57d5b3799973ccb6cecb2f78e76695b5',
        'T-1.csv': '4f4d7416a1546e8023c573594959a072',
        'T-2.csv': '38820b15d9d59b6a84957183bf129e63',
        'T-3.csv': '3b5cd76451744e5c4e78c05e45fc6df8',
    }

    def __init__(self, root: str, download: bool = False):
        super(SMAPDataset, self).__init__()
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
