"""
@Time    : 2021/10/24 1:11
@File    : yahoo.py
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


class YahooDataset(TimeSeriesDataset):
    __subsets = ['A1Benchmark', 'A2Benchmark', 'A3Benchmark', 'A4Benchmark']
    __file_list = {
        'A1Benchmark': {
            'real_1.csv': '4b06fc44bb265a0f2465f772553c230c', 'real_10.csv': '5e8091b0197125cd77bd4d9d8cc57c76',
            'real_11.csv': 'f62766a6746a4d5446615000ba09dc74', 'real_12.csv': '5b9ff758b733276161ed62743c0e8bf6',
            'real_13.csv': 'c11a1c8f7db70fa868e164349a930bd3', 'real_14.csv': '58a46343e43384fdc5319e675e81f75f',
            'real_15.csv': '126df75c8805a649f6f9a0116e5a4ac1', 'real_16.csv': '969b69310c5d6ed6f971d6c96a13d2c3',
            'real_17.csv': '20d2a541aef1bbe484fc83e1fa945de8', 'real_18.csv': 'f3d064e9d76d474018871e85b38c7e41',
            'real_19.csv': '207e332e6605360f5b1447464d34e212', 'real_2.csv': '01394e712242d3f576b46b6a24d3d5ca',
            'real_20.csv': 'e79cd48bee48d73a4a5d6787a802c057', 'real_21.csv': '8e859ab11b3fed2351103cd45e418b9c',
            'real_22.csv': '540ada64acdd238c8e9757956f596848', 'real_23.csv': '18488f4c714d66658a4f8c4132b7cca5',
            'real_24.csv': '572a0842290c395dffca08c3cd17adc1', 'real_25.csv': '370900b15654baa6a6f02c99af9214e7',
            'real_26.csv': 'e9b8843f665e39c758efb627a40cc851', 'real_27.csv': '16fdb149061199a4bb681a27be81010f',
            'real_28.csv': '2dd895242f9a3812544f94d46f5f3989', 'real_29.csv': '0ec6224d17a12ce066ded102d3b93170',
            'real_3.csv': '128a02a3455547a25b7e8cf7e4800249', 'real_30.csv': '66394013a267a6b99f80fdc12b148c96',
            'real_31.csv': '7c0b384bff57e2160b419546fa9d3a1a', 'real_32.csv': 'e8c9f630ba0579b41af905a3fadee6cc',
            'real_33.csv': '8790ddce347832f1e59619ea050f1ecb', 'real_34.csv': 'efde5ff635a75f6d105962370169ffa7',
            'real_35.csv': '593c00ef9b124e96e647d8a766a683ae', 'real_36.csv': '45cb3649b722318116603b2b92ed13ad',
            'real_37.csv': 'e2c21dedddfc4cacf95a906ee9aa1876', 'real_38.csv': '02abf6cecdf2adaa4c05e51b4725a50f',
            'real_39.csv': 'ca6640d9c07bbabd0d32384bb1347846', 'real_4.csv': '1a1ac10a0959de98c1ebb5771c2dd706',
            'real_40.csv': 'b911a6f01d1eb7264e3c0462e6fe2fdc', 'real_41.csv': '73d0bf7c706acfa050088856ab030843',
            'real_42.csv': 'b6655d310f545965bcb6f755eba9360f', 'real_43.csv': 'fa25f3675f8b52be182bed6648146961',
            'real_44.csv': '91b80cc10dec600378b9de2d81524d79', 'real_45.csv': 'd96e0b5e47d2cacafad25e63e71168ba',
            'real_46.csv': '9abf1ef74239b7bab5f00f95698ddd63', 'real_47.csv': '537bf6e5e1b83ec25099b8d02a4a398f',
            'real_48.csv': '7548719848f88c360371861163652992', 'real_49.csv': 'f3d064e9d76d474018871e85b38c7e41',
            'real_5.csv': '75f9e2caba307ec2f04c9983a479b5eb', 'real_50.csv': '22d744f53b0dcdc3416d46b07ba15938',
            'real_51.csv': '384149ae11fd7814ab393893c7b572d1', 'real_52.csv': '77d9c02719ffaccd1ee9c35c75b7e58d',
            'real_53.csv': '92fd4476afde472c158c4d5f2ddd51e6', 'real_54.csv': '2e926007ba389846c97968375b7e4cd0',
            'real_55.csv': 'ac46affa00c73fdfdf347d9e8fe476c4', 'real_56.csv': 'e7b7aa60b219e2bb67b3cda1785c3a91',
            'real_57.csv': 'fe501f791159fc39676f66b9cacbc979', 'real_58.csv': 'ef89c8406cd6eb3ca61a51687d62cd34',
            'real_59.csv': 'b4ace0e84d3e3dad2bc515d0d97f8d6f', 'real_6.csv': '955b73913d203b164b7e0c28fea972d0',
            'real_60.csv': '9f448b2458d8c4d2a0767f8d30eea6d5', 'real_61.csv': '8b138374d76d841bc700f80f98eaf545',
            'real_62.csv': 'eb63a0677444dedc990801000b755b5c', 'real_63.csv': '56d3c0d1736548a397d50663c20197e9',
            'real_64.csv': '18a2ec49c113dfdc62904be87c0a0f60', 'real_65.csv': 'dcb08f7fe6a6f4110367b77f56ac8e93',
            'real_66.csv': '332da8b909bd8843fc9335e83d6de263', 'real_67.csv': 'ba63a44dbd8fd6d7c018372fb85c7d74',
            'real_7.csv': '5030e08fe659f98e3f12f0f728fa6e28', 'real_8.csv': '11f3db008b763e8668bdcebe6df2bab1',
            'real_9.csv': 'a82b98094e45b125a47f590f13231c50'
        },
        'A2Benchmark': {
            'synthetic_1.csv': '1733044263da51a57913b2945d1a4189',
            'synthetic_10.csv': '5a5043faf420c37f9ed623abd3bf9a6e',
            'synthetic_100.csv': '398d42d24efb5099ee83a187b9945ab3',
            'synthetic_11.csv': '0e2cbdf45c841157fbe087ef6394235f',
            'synthetic_12.csv': 'c452b41b3ba343e77f8e0f8b83fa2268',
            'synthetic_13.csv': 'aa7d7a84291cbb58fb9bdb96ec975e78',
            'synthetic_14.csv': '912c056d6bbe10ffb19905a4b0adcf45',
            'synthetic_15.csv': '800c1d5d8ae86a3016781e0276688122',
            'synthetic_16.csv': '1d8200836709f732f1d29c4ff3dfd312',
            'synthetic_17.csv': 'd236c7952f5b3658fa566de9564ff137',
            'synthetic_18.csv': 'fda95af8919ac6baed84031f0f75c0dc',
            'synthetic_19.csv': '0b1e7a0c36e7414d284a084d343a9dc2',
            'synthetic_2.csv': 'aaeb00f1df2bb9d63845ca6e20b14fe3',
            'synthetic_20.csv': 'ca59e2fe9310236674903ea61479708b',
            'synthetic_21.csv': 'cf25fbeed5c29f6ce9ac8946098b638b',
            'synthetic_22.csv': 'cb4d30542c86886f0c627f71b6333a9f',
            'synthetic_23.csv': 'f581fce2951ebd4b7a7be0186d97b4fa',
            'synthetic_24.csv': '1e54a349822b0a527780f78f8a24ddb4',
            'synthetic_25.csv': 'd66450a3bad5b78e15c7a60a9e8901c2',
            'synthetic_26.csv': '8dec308fd34f9fabbab57bc7d9faf3da',
            'synthetic_27.csv': 'b4b93461b8d328475e0e3cfd87807abe',
            'synthetic_28.csv': 'bae7effd99b11c609f6c26708ead8e8a',
            'synthetic_29.csv': 'e5e603889e6990744e827b6b12ce96be',
            'synthetic_3.csv': 'cdd3cf14f9a500391e357725ee2b99e0',
            'synthetic_30.csv': 'c2800332e865fec536ac7a5f0ef8b5bb',
            'synthetic_31.csv': '2f3768adf62fad6ec4783802ec8ec6a0',
            'synthetic_32.csv': '5145f4b96a1d70197cc355281693ecec',
            'synthetic_33.csv': '5ec3297875765c6380ab27d391cd1b2c',
            'synthetic_34.csv': 'cd1de7d725d55c2f3deb2d9bf8242937',
            'synthetic_35.csv': '59bc294751a0c9f32e04c7ac6277f47d',
            'synthetic_36.csv': 'f55373eefb4bbb27ebec8de6b7c957c7',
            'synthetic_37.csv': 'a333ee069d0582d728778b723398d0c9',
            'synthetic_38.csv': '923dfce23890163289eebce3bc95fcfb',
            'synthetic_39.csv': '722d0aa52bd550666c170f546b9b70bf',
            'synthetic_4.csv': '9ec7317e22463b9da94c1d18e9b5c4b6',
            'synthetic_40.csv': '8ca7bd5f4a659ff522c400d9eb91f964',
            'synthetic_41.csv': 'e70e632b40a5ce2723bc23c80a7ee61c',
            'synthetic_42.csv': 'dfee924dbdd9c54862b051a0e73ccd4a',
            'synthetic_43.csv': 'c6d97c7a4ef0755614d5af65ef7e177d',
            'synthetic_44.csv': 'f7c25ca31b16dd8dbc00cb7d981e3002',
            'synthetic_45.csv': '36a7f06ff866fd4e187c9583a47ad08a',
            'synthetic_46.csv': '4253fbf4cb550d27abf3bf3ae50b79ed',
            'synthetic_47.csv': '6428e7cacd1d37a11f9e94f1181dbe3b',
            'synthetic_48.csv': 'e809198db768b51356712a179b25a9ba',
            'synthetic_49.csv': '59d83ad3d6a80964a80952ab2d121adf',
            'synthetic_5.csv': '95de2f3b8f4f924754a0c425c7f71c92',
            'synthetic_50.csv': '3a16711d50e264cb0dbd7d7c36f9284a',
            'synthetic_51.csv': '2b806fec81a5a9cbe4e36118fe735c6d',
            'synthetic_52.csv': 'cf7fc80b4366e6b89a74c3b4d7cc2cbe',
            'synthetic_53.csv': '9e0a3670e01d991b6f8666a0596ae84f',
            'synthetic_54.csv': '6bfeea1e882732ad866414681943d44a',
            'synthetic_55.csv': '68b3512db45aa3c815a83fb0d1178f06',
            'synthetic_56.csv': '43fc4fa67dc13dfccf562e6d20239a96',
            'synthetic_57.csv': '6e8d9c1ccfa43a7676520d57d9e62a26',
            'synthetic_58.csv': '1378a5488b24d193536d83fad43a83a7',
            'synthetic_59.csv': '72ca0e1a0c2b3b4dee7ada54680c322b',
            'synthetic_6.csv': 'f1bf48a8eec684a3984c549edc46758a',
            'synthetic_60.csv': '6c46d9e1c71512be5db358f9b529a0c8',
            'synthetic_61.csv': '97a786f26870ffe367decf8fecac9977',
            'synthetic_62.csv': '950510084e23a6e996beff6a5757e410',
            'synthetic_63.csv': 'b18bb2a0101db4e4833af0fd54a3f7c4',
            'synthetic_64.csv': 'b2ee40e0f0faf7292429fbb4c3183387',
            'synthetic_65.csv': 'e71c57627a355ae7b647cbc8beeb28a3',
            'synthetic_66.csv': '7bc9047a64df2a987343ec15028a7398',
            'synthetic_67.csv': 'f063b564ad3c094ebb2abf9b580489b2',
            'synthetic_68.csv': 'daa62662118ebdb6fc8eb467fe1d63b3',
            'synthetic_69.csv': '36c30d6f3eed49efc7107bdd9099709a',
            'synthetic_7.csv': 'b69d8243a65194cfaedbfdecb1cab5bb',
            'synthetic_70.csv': 'd4dd981a15da611d23e15894350ec73a',
            'synthetic_71.csv': '4a46b4538f7bd67422870d4ba2b2bace',
            'synthetic_72.csv': '79d937100f4af53089353a405d5df57b',
            'synthetic_73.csv': 'c3ca6d3e40eb1d62082efbff720e5629',
            'synthetic_74.csv': '971bc2e66ce46ba5d9d27a6c15449fd9',
            'synthetic_75.csv': 'c1c41465a1add699ce707c3917e1068c',
            'synthetic_76.csv': 'f5ff6fb589e36db078f1d2122015fec8',
            'synthetic_77.csv': '04467aadcb7e479376ef47dc62ec8c8c',
            'synthetic_78.csv': '6d89201583d2e0a7475f67de7f97b514',
            'synthetic_79.csv': '38a27e750729eeef2070693fa62af798',
            'synthetic_8.csv': '2e5015b03e22f09281d75e73dc55b84e',
            'synthetic_80.csv': '37acd1a5b93a6a3709bc63e3320af8f2',
            'synthetic_81.csv': '8222a989be42ead25e41ae7c23a19755',
            'synthetic_82.csv': '7ecb4d8f6aa8fa5c65075a9333b88758',
            'synthetic_83.csv': 'fa0e5c0ed28c5a7478b0c690e7a439cd',
            'synthetic_84.csv': '8a6cdf009fb6221888c869f1b2f3b375',
            'synthetic_85.csv': '9cc69ce4929720a842d1eca18f2a41fb',
            'synthetic_86.csv': '0801775943f1fbdac6ca9a4c8fdb3559',
            'synthetic_87.csv': '764d7dce7aece41357c8b1f4e78565b8',
            'synthetic_88.csv': '169cc1519e1810d4bbc1dfb06b01e392',
            'synthetic_89.csv': '402f580a4eb0d67868cf9c82b8fe859b',
            'synthetic_9.csv': '2b38b0842d9ddb33c8968ff20ca09e8c',
            'synthetic_90.csv': '699eda8a4e701b45f839b7b23d62664b',
            'synthetic_91.csv': '366e71f0dd6d9d6e5951bc2c3a1abe19',
            'synthetic_92.csv': '8d93e784b1d91765629d31f5d48e85d7',
            'synthetic_93.csv': '768c87710ff43b3129aead888c984265',
            'synthetic_94.csv': '921003b701e328841c0b43622e2c0d49',
            'synthetic_95.csv': 'b136edbbf63d76846d76a728d1b06c01',
            'synthetic_96.csv': 'dc86c0768e8b51edfe753e722b688b34',
            'synthetic_97.csv': '7ee42049fbf834c7ff287d902b0fc435',
            'synthetic_98.csv': 'df294dc8775ef6c46df984840d0db8f3',
            'synthetic_99.csv': '8a0239b38c1b7c5fbdaf63854ab385d4'
        },
        'A3Benchmark': {
            'A3Benchmark-TS1.csv': '2fe5715e88b6a3dd551299cdad017dc1',
            'A3Benchmark-TS10.csv': '88441d8f5e1e1110ed1a172d0bbbde26',
            'A3Benchmark-TS100.csv': 'df1a97f9d65c91abccbe6eb1072df936',
            'A3Benchmark-TS11.csv': 'c30732cfcea7eb9940afd7ef0a885957',
            'A3Benchmark-TS12.csv': '7f71868d4efde3f143352b86e51cfd7e',
            'A3Benchmark-TS13.csv': '6d73dae685bd76a34d36ffa5cc5436e0',
            'A3Benchmark-TS14.csv': '215098f0595445472d85aefd53abf0cc',
            'A3Benchmark-TS15.csv': 'd55b3f7cc56090336b30dade1cbea913',
            'A3Benchmark-TS16.csv': 'dcb3c1d6e8c998a24e86bbffd4ee86fc',
            'A3Benchmark-TS17.csv': 'a997ea31ee3e6b3289c262fa5d819c5e',
            'A3Benchmark-TS18.csv': '496077af9f822162befb35869c238e46',
            'A3Benchmark-TS19.csv': 'ac2dc911dc67733caed3234164251b58',
            'A3Benchmark-TS2.csv': '0cb1c9d48612fefa671b51dbd6efec14',
            'A3Benchmark-TS20.csv': 'f9bd7adce83267d437d3d9cb6baba61e',
            'A3Benchmark-TS21.csv': '8500a2530cd2186d92b897ec0eb2235e',
            'A3Benchmark-TS22.csv': '5f04a89c319684cf67cb4f38167aad92',
            'A3Benchmark-TS23.csv': 'b600fb24d2811459d88f5695f999c878',
            'A3Benchmark-TS24.csv': '24c4b0ca2bdd4fa49d0c31ba4eb273ef',
            'A3Benchmark-TS25.csv': 'e4a2aab508b2af5b983367de84be7835',
            'A3Benchmark-TS26.csv': 'cbd2571e28b9362441dcd305c49bfc82',
            'A3Benchmark-TS27.csv': '6d39854a2a64c52016944aefa4bb8b58',
            'A3Benchmark-TS28.csv': '1af28981bd79d7d128c46a364e8d5ca4',
            'A3Benchmark-TS29.csv': '39560470825ce31308f8e1fe00800595',
            'A3Benchmark-TS3.csv': 'f269b1d50958ee93059fea2ecf9e496a',
            'A3Benchmark-TS30.csv': '237f6e06d96914f436a067df936c2432',
            'A3Benchmark-TS31.csv': '52397107d1d647378245b3337f53749f',
            'A3Benchmark-TS32.csv': 'b0fc3befd4fd031dfb9b770f98642265',
            'A3Benchmark-TS33.csv': '51a0609ec1a6a858d03bd3b65b110d8b',
            'A3Benchmark-TS34.csv': 'b61fffdd0cbe37e9ba449517dc83127a',
            'A3Benchmark-TS35.csv': '5589c4ed85711a0b04ffc8304e47f743',
            'A3Benchmark-TS36.csv': '685eb31bb5aafe203824d1db76f4e39f',
            'A3Benchmark-TS37.csv': 'ce957d33f969154ea7b8e62712e93c83',
            'A3Benchmark-TS38.csv': '4887162564b09fab542e1df58566823c',
            'A3Benchmark-TS39.csv': 'be840e877101ee07055f0c3f679c2286',
            'A3Benchmark-TS4.csv': 'ded9e0f3edf37c27746c3f1917c991c6',
            'A3Benchmark-TS40.csv': '088a3da257ac29a5d90976e601aa9bc3',
            'A3Benchmark-TS41.csv': '6686937d9045d937d9f940eec08b8f97',
            'A3Benchmark-TS42.csv': '3860184a0665979f693b3d809b6680fe',
            'A3Benchmark-TS43.csv': '09cda2b5450ae9e1f140224f4601dfc3',
            'A3Benchmark-TS44.csv': '36586dca4e9b63527e56717648cc961a',
            'A3Benchmark-TS45.csv': '0af4e03b53f4f60f31d3577622d08cf2',
            'A3Benchmark-TS46.csv': 'd03c4e14a683bd107cc6d88d8d71ea4c',
            'A3Benchmark-TS47.csv': '72b903c1475bd8c2315add8706206e4a',
            'A3Benchmark-TS48.csv': 'cb39ccd284923fe013c17c13cc05f261',
            'A3Benchmark-TS49.csv': '1dcbd382ffba872a9b3d1f3e8746ba97',
            'A3Benchmark-TS5.csv': '3dcc17df1b8b157b445e6732325020d7',
            'A3Benchmark-TS50.csv': '0544d524201e1b7d23a6ad8df7d83a43',
            'A3Benchmark-TS51.csv': '4f0e48ffec7f584011a76accaf867b8e',
            'A3Benchmark-TS52.csv': 'e3588a83f79698dcc65b4ce1bd493418',
            'A3Benchmark-TS53.csv': '4607ab74c4306d9bad732f37c51d7fff',
            'A3Benchmark-TS54.csv': 'a403e672548d9e967a024be96b18e267',
            'A3Benchmark-TS55.csv': '1883eba8d337003af794a3cf8fd6a364',
            'A3Benchmark-TS56.csv': 'ceef28f5fbc5248073923a17d821cb85',
            'A3Benchmark-TS57.csv': '8f320dc1c75a2ce33e9f52bd01a5589d',
            'A3Benchmark-TS58.csv': '29cca7f372a43d828945e6ba806b93b9',
            'A3Benchmark-TS59.csv': '7520792b96c90d6b5e80deba4d0261a1',
            'A3Benchmark-TS6.csv': '041c93fc84a26ba3f5edab758c3c7f21',
            'A3Benchmark-TS60.csv': '830021ce6d111c22f2ec23a45b156d23',
            'A3Benchmark-TS61.csv': '70acb2e9c7b2006738e04c10d42610d5',
            'A3Benchmark-TS62.csv': 'ef23afe4875a2df666f9fae9a5b3a72b',
            'A3Benchmark-TS63.csv': 'a8f851f39fa8445928f3af03239e00c5',
            'A3Benchmark-TS64.csv': 'f32bdf749034876ebaf8f079a88d94bb',
            'A3Benchmark-TS65.csv': '48b4196570e27a5260465b6a5ce8328a',
            'A3Benchmark-TS66.csv': '32afcb3d539a31a86e6a64c4866334f1',
            'A3Benchmark-TS67.csv': '4358bb2a573625302091fea2aad1f021',
            'A3Benchmark-TS68.csv': 'b3475776c9d4d1f7f75df0616e6e48be',
            'A3Benchmark-TS69.csv': '8bfe17614f374fbbeaf4215313b5c2d6',
            'A3Benchmark-TS7.csv': '881dfa03f83bd3018c1853249d139e32',
            'A3Benchmark-TS70.csv': 'daf95e165bab6f75aa3dc046b89baf05',
            'A3Benchmark-TS71.csv': 'f7f3bca0e45063a28a00b60e4b4a1a46',
            'A3Benchmark-TS72.csv': 'f3f2361edecaca8cf5c744d49c34a67d',
            'A3Benchmark-TS73.csv': '8e79f26c107452291a1f45bccad399bc',
            'A3Benchmark-TS74.csv': '7c88fd24616c9918553782431a5d5304',
            'A3Benchmark-TS75.csv': 'a2a7ebc830d65d0d146bb2cf59a14f49',
            'A3Benchmark-TS76.csv': '91523c3e4509329b593cc10c9488f0cd',
            'A3Benchmark-TS77.csv': '605053464c71750307a0d64712523109',
            'A3Benchmark-TS78.csv': 'ddfd87a8fec0e1c664d64db0f53ea384',
            'A3Benchmark-TS79.csv': 'e26c6098ce607bde03f3f852da6f24f1',
            'A3Benchmark-TS8.csv': 'ee345b49aa815dd918a5ad29a374ca4f',
            'A3Benchmark-TS80.csv': '5abec11bc2f6c530f8e1330d926ccdfb',
            'A3Benchmark-TS81.csv': '2a4c52d12cc6decd643128557e856b93',
            'A3Benchmark-TS82.csv': '8785584c0b07272a945cc8b6153553ab',
            'A3Benchmark-TS83.csv': '82d0808c5ea47ab2c7bd586b9b794b38',
            'A3Benchmark-TS84.csv': '75c5fef2bbf7bcf541f5bd183f8ed968',
            'A3Benchmark-TS85.csv': 'af00318cdb0cbc6751c143249845f96e',
            'A3Benchmark-TS86.csv': '929ac93dd0a6bc7e8c46589dc771d719',
            'A3Benchmark-TS87.csv': 'e0d342ce45c2e96212120b6690a8c1ef',
            'A3Benchmark-TS88.csv': '2397920daf43f6f4e60c224e6ee499f0',
            'A3Benchmark-TS89.csv': 'c7747ba0ddcf49e856b1e754acc1bacb',
            'A3Benchmark-TS9.csv': '2baadd9bf8efccf3eb02e92252318d5a',
            'A3Benchmark-TS90.csv': '4a3a00c9aa7390b0d916ab141e6fbfa4',
            'A3Benchmark-TS91.csv': 'c37639df0d13bedd4fb6593b67b285b6',
            'A3Benchmark-TS92.csv': '4cc69feb11372683283bb34405b5dc29',
            'A3Benchmark-TS93.csv': 'a386acfd69cdf8fa68affe7d4d3a2fe9',
            'A3Benchmark-TS94.csv': '443a485e708f1716c6104b8ace0a613b',
            'A3Benchmark-TS95.csv': 'be96462ec40cd95d72142b29cba391e1',
            'A3Benchmark-TS96.csv': '610b6024f9911241884b9e71ee43aa2b',
            'A3Benchmark-TS97.csv': '27c224a405bbb8031130e39fa50010f1',
            'A3Benchmark-TS98.csv': 'f48da3a8ff7b6a6d45e7ccd8f7e91fc8',
            'A3Benchmark-TS99.csv': '29bc3277168e9196e841a610791b97f3',
            'A3Benchmark_all.csv': '3e8346e13af67e0b7ad99853ed9e7f21'
        },
        'A4Benchmark': {
            'A4Benchmark-TS1.csv': 'da270f78f0ed0617c4437f87700156ff',
            'A4Benchmark-TS10.csv': '528a1a674f163dff8ba10ff8ddb64b53',
            'A4Benchmark-TS100.csv': '50d32616a8744238407daf1f9e5558c6',
            'A4Benchmark-TS11.csv': 'ab0569d8962fa31b125b3d04f89b7f80',
            'A4Benchmark-TS12.csv': 'd0e881b7f3ec8da4bf8142a116f0498c',
            'A4Benchmark-TS13.csv': 'd714833c02ae30cf351961b5ce5c53d2',
            'A4Benchmark-TS14.csv': '1ad817b214fb13d3736c3da5eebfc9d5',
            'A4Benchmark-TS15.csv': '488d47f7607774d5dabb908e7d767a41',
            'A4Benchmark-TS16.csv': 'a08fc26cf5c5491cfea2161f732a6866',
            'A4Benchmark-TS17.csv': '17481e2d7c6903d1706ba7abcaa19f9f',
            'A4Benchmark-TS18.csv': '60bd29946b4ea2d6a0b5979ca4b7305e',
            'A4Benchmark-TS19.csv': '3abc5fe907c250dae87505c798f3aa9d',
            'A4Benchmark-TS2.csv': 'f15c9d538d20f1d6d7f9265095c921f6',
            'A4Benchmark-TS20.csv': 'c0a06f77f8e7b2e23d59ec50fdb82a8b',
            'A4Benchmark-TS21.csv': 'f266c569de5925473e1d004f760644fc',
            'A4Benchmark-TS22.csv': '2028ec7e8fbe658104e882b099a4b8a7',
            'A4Benchmark-TS23.csv': '135732970dce74e4eba5b559725bdfb7',
            'A4Benchmark-TS24.csv': 'dcf92408e18933829f5bf33b12736148',
            'A4Benchmark-TS25.csv': 'e1bf2192879ef41142c9a9089fab7a6e',
            'A4Benchmark-TS26.csv': 'e8e0636855c49ab06e572c0cbc1a6f81',
            'A4Benchmark-TS27.csv': 'e18ed557b4ae931238354076860786ab',
            'A4Benchmark-TS28.csv': '46be1bc882094423fd74a920af86f9b3',
            'A4Benchmark-TS29.csv': '14366d859e3e575907b7e5b7affe1664',
            'A4Benchmark-TS3.csv': '07685f1c23a11a5a7364ac99acd47a90',
            'A4Benchmark-TS30.csv': '68d243596419c803fd6c709b6649c7b5',
            'A4Benchmark-TS31.csv': '1e8290c58296de65533bfe507233265f',
            'A4Benchmark-TS32.csv': 'f51336138b03f047b853602d2b05d850',
            'A4Benchmark-TS33.csv': '26415987dbe46e41e7a739045a06c2a6',
            'A4Benchmark-TS34.csv': 'd4dcb9015d93441c3e0bb58679d17256',
            'A4Benchmark-TS35.csv': 'e5623f0696bdc328805d58e9ad82984e',
            'A4Benchmark-TS36.csv': '191291ce5e0917431c25ea0b52d57adf',
            'A4Benchmark-TS37.csv': 'ed1682195fe441692ba70f07c2fb0b5d',
            'A4Benchmark-TS38.csv': '595e985e60f293cb1de75b8934a72437',
            'A4Benchmark-TS39.csv': 'dc3edb35bc8c37dab8373c0af916e22e',
            'A4Benchmark-TS4.csv': 'ded9e0f3edf37c27746c3f1917c991c6',
            'A4Benchmark-TS40.csv': 'ed333b117fb23017f1425a21dccf1cd2',
            'A4Benchmark-TS41.csv': '07cb03562d2cb578d4005aa653e32435',
            'A4Benchmark-TS42.csv': '88b89adff0adbf1361951cb2b1cd4452',
            'A4Benchmark-TS43.csv': '767de793eb435b2e5b80d03525477f2f',
            'A4Benchmark-TS44.csv': '3c7ae08e60491f74c3a2089a371e278f',
            'A4Benchmark-TS45.csv': '13af754f3372f5605d3f504d829c998a',
            'A4Benchmark-TS46.csv': 'd8d7870fca0f08d6ae4cfe38d364dfe9',
            'A4Benchmark-TS47.csv': '538ee903732067105e01c59785897f51',
            'A4Benchmark-TS48.csv': '76d1c8a535ef72a9a1a5f9a4396b45c3',
            'A4Benchmark-TS49.csv': '0bfa73f5acbddbfd0fd237dd9f7c50d3',
            'A4Benchmark-TS5.csv': 'abc975308ea5745cba0c45f8d08068ad',
            'A4Benchmark-TS50.csv': '5085511bd69b5eaee0f6d584252d6525',
            'A4Benchmark-TS51.csv': '2cb5f0e6ecf0a4ed82e8a6e1fb9e4987',
            'A4Benchmark-TS52.csv': 'dd5d3e6dc12cbae6bfd9a1bf87fd88ed',
            'A4Benchmark-TS53.csv': '00ef12ffeb3ea6c1f15bfb2644023ddc',
            'A4Benchmark-TS54.csv': '10adf7fea4a43bc357288d7607ec72ed',
            'A4Benchmark-TS55.csv': '648e25a9e59ecbb56a61b9a999c279f0',
            'A4Benchmark-TS56.csv': 'b3e58f7e4189ef309d2a96fcb3e94655',
            'A4Benchmark-TS57.csv': '61e25a85b6eb591137558819c2c23e14',
            'A4Benchmark-TS58.csv': 'bcbce5c3f63f6b911b2f3747e1dd5383',
            'A4Benchmark-TS59.csv': '1640db1b18cad377764a29846d973e67',
            'A4Benchmark-TS6.csv': '5199dd47bc0456d3423d472a538f4579',
            'A4Benchmark-TS60.csv': 'f6966f1301fa9e54a86636ecbdd2155b',
            'A4Benchmark-TS61.csv': '4942a425aa383d04eae48f0d3b99bfdb',
            'A4Benchmark-TS62.csv': '78dd7d1875f8a1c9e9267735f30fbdcc',
            'A4Benchmark-TS63.csv': '61e60e68c720125dd972ed0b0d28fe64',
            'A4Benchmark-TS64.csv': '4fe42b90535f01a83b9a9005ebdaf77b',
            'A4Benchmark-TS65.csv': '7b0c3cc57b8223618c1e0d33d0c6a840',
            'A4Benchmark-TS66.csv': '7084b107e6989e1964bcf5431ab3011d',
            'A4Benchmark-TS67.csv': 'd73013ffa59b4c93488f6046bc8ae236',
            'A4Benchmark-TS68.csv': '4b8c368bf8ea3abdde5fcad83036b5ca',
            'A4Benchmark-TS69.csv': '9fb0df5f70bca4f0ccf04f526c2f8fd5',
            'A4Benchmark-TS7.csv': 'e7e1c93211971b74d862ed70ba01dff1',
            'A4Benchmark-TS70.csv': '9c5816024abf2406142011ee01414da9',
            'A4Benchmark-TS71.csv': 'f1640bbd8a8178f908552b8d6404b474',
            'A4Benchmark-TS72.csv': 'f3ba9a4cc02ef07e548c2afd7eedf02b',
            'A4Benchmark-TS73.csv': '3e79082f613a00ddb64f02be9c1426c0',
            'A4Benchmark-TS74.csv': '5d7bcf97b434e4aea0f99d785385896f',
            'A4Benchmark-TS75.csv': 'b17faa5a5c819c5f9de7d18d47937317',
            'A4Benchmark-TS76.csv': '26edfebd09bdcc4d82bab0c7ae357343',
            'A4Benchmark-TS77.csv': '6501e1225db662b3f600209437cc53df',
            'A4Benchmark-TS78.csv': '294e74997b40068d066be7d3d7b78f89',
            'A4Benchmark-TS79.csv': 'b3b1b76b8f1c316680470151d98f8cd4',
            'A4Benchmark-TS8.csv': 'aa93a1a3b6efbc25fcbdf53e67874029',
            'A4Benchmark-TS80.csv': '54641f1bce61d1de89c5c9ba4726bfe3',
            'A4Benchmark-TS81.csv': '58b33041116e654cdc6d93cb7cc50974',
            'A4Benchmark-TS82.csv': '9c588be83bacedf9429c43e553a180db',
            'A4Benchmark-TS83.csv': 'cdff881909d6ca1f625f4608accdcea7',
            'A4Benchmark-TS84.csv': 'a02ab6ec6942295a95c3d87ea1bd93ae',
            'A4Benchmark-TS85.csv': '082b7d4e3c4b62bfe21faea307db0f87',
            'A4Benchmark-TS86.csv': '0d8fa76935c8c0a732e74938c0bf6f8c',
            'A4Benchmark-TS87.csv': 'e90858c24b16215063b1e35970a3d2eb',
            'A4Benchmark-TS88.csv': '1f249666cf79a86a6816a5ca6210abeb',
            'A4Benchmark-TS89.csv': 'f826473efb1518dc462713d52c6b95b2',
            'A4Benchmark-TS9.csv': '8ae9f68e8b5ffe2b59ef319c090c80ed',
            'A4Benchmark-TS90.csv': '758f4910a54b448c1693b417fdd730d9',
            'A4Benchmark-TS91.csv': '3a212180c4397bc3139f817c664aabd4',
            'A4Benchmark-TS92.csv': 'cbcf974ce0bfcfe8ea74593c92ae1907',
            'A4Benchmark-TS93.csv': '5524669eeec5a19e17362882e8acff70',
            'A4Benchmark-TS94.csv': 'b976dbee0b214c71148940b2babb8167',
            'A4Benchmark-TS95.csv': '62283939b0e7853c2bc7e5b4496c6811',
            'A4Benchmark-TS96.csv': '6c3e0dc91ffb55d4ade4fe1ded2d564a',
            'A4Benchmark-TS97.csv': '188b22e5ccd54d636b0c2fe6e522a584',
            'A4Benchmark-TS98.csv': '26db7cb1bff963deb06d09d60d31e9b3',
            'A4Benchmark-TS99.csv': '1095645332cf2185adb8da643bd02cbe',
            'A4Benchmark_all.csv': '703b0d23c74d09ab6a34242b4fd76e80'
        }
    }

    def __init__(self, root: str = None, subset: str = None, download: bool = False):

        assert subset in self.__subsets

        if root is None:
            root_path = Path.home() / 'yahoo'
            warnings.warn(
                f'The `root` path of the dataset is not set, using user home dir {str(root_path)} as default.')
        else:
            root_path = Path(root)

        if download:
            raise ValueError(f'The Yahoo dataset should be downloaded manually. '
                             f'Please download the Yahoo S5 dataset at '
                             f'`https://webscope.sandbox.yahoo.com/catalog.php?datatype=s`!')
        else:
            assert self.__check_integrity(root)

        data = []
        labels = []
        timestamps = []

        for csv_file in (root_path / subset).glob('*.csv'):
            df = pd.read_csv(csv_file, quotechar='"')
            if subset in ['A1Benchmark', 'A2Benchmark']:
                df.sort_values(by='timestamp', inplace=True)
                label = df['is_anomaly'].values.reshape(-1)
                timestamp = df['timestamp'].values
                df.drop(columns=['timestamp', 'is_anomaly'], inplace=True)
                data_item = df.values.reshape(-1, 1)
            else:
                if 'A3Benchmark_all' in str(csv_file) or 'A4Benchmark_all' in str(csv_file):
                    continue
                df.sort_values(by='timestamps', inplace=True)
                label = np.logical_or(df['anomaly'].values, df['changepoint'].values).astype(np.long).reshape(-1)
                data_item = df['value'].values.reshape(-1, 1)
                timestamp = df['timestamps'].values

            data.append(data_item)
            labels.append(label)
            timestamps.append(timestamp)

        super(YahooDataset, self).__init__(data_list=data, label_list=labels, timestamp_list=timestamps)

    def __check_integrity(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

        for category, category_dict in self.__file_list.items():
            for file_name, file_md5 in category_dict.items():
                if not check_existence(root / category / file_name, file_md5):
                    return False

        return True
