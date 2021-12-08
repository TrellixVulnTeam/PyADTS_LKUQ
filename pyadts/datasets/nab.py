"""
@Time    : 2021/10/18 17:59
@File    : nab.py
@Software: PyCharm
@Desc    :
"""
import json
import shutil
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from pyadts.generic import TimeSeriesDataset
from pyadts.utils.io import check_existence, download_link, decompress_file


class NABDataset(TimeSeriesDataset):
    __link = 'https://github.com/numenta/NAB/archive/refs/tags/v1.1.tar.gz'
    __filename = 'v1.1.tar.gz'
    __label_filename = 'combined_windows.json'
    __label_md5 = 'd2ec0ef3652beca9e4847fe4f3815529'
    __subsets = ['artificialNoAnomaly', 'artificialWithAnomaly', 'realAdExchange', 'realAWSCloudwatch',
                 'realKnownCause', 'realTraffic', 'realTweets']
    __file_list = {
        'artificialNoAnomaly': {
            'art_daily_no_noise.csv': '2a972131fc973ce3bf657a7646424122',
            'art_daily_perfect_square_wave.csv': '7204f333e0bf3c08dc4e39e7eaf542e6',
            'art_daily_small_noise.csv': '415a45c424a0a893bdc7089709cb3ae3',
            'art_flatline.csv': '3cbbec0d3c1a0450414b52077f474c53',
            'art_noisy.csv': 'f36b0ac2c09c78c7724b6846c660bbbe'
        },
        'artificialWithAnomaly': {
            'art_daily_flatmiddle.csv': '50627637feb00e11fb5c429c1e2692eb',
            'art_daily_jumpsdown.csv': 'e9ca9d20a32d5553b7e83ea9f3689369',
            'art_daily_jumpsup.csv': '295288850f680beef1c0a42a57d07167',
            'art_daily_nojump.csv': 'e0217a357ef5f5dc8383fa52a9fa9c0e',
            'art_increase_spike_density.csv': '8c3f32e6fcf8bb19690dfaccbeca252a',
            'art_load_balancer_spikes.csv': '2bfee3910d2acff2600df30f23d009d1'
        },
        'realAdExchange': {
            'exchange-2_cpc_results.csv': '5709999ae3a64c598f01a1a94b8fff2b',
            'exchange-2_cpm_results.csv': '0ed20905ff8a57d1643ed1366a95eaf8',
            'exchange-3_cpc_results.csv': 'f0577f6adfce2b8e26c81d6e7d548011',
            'exchange-3_cpm_results.csv': 'd2efba8ed2f0206408e861383dbb7f03',
            'exchange-4_cpc_results.csv': '8adfb571860d3a9fcb9c89904011d477',
            'exchange-4_cpm_results.csv': 'db2fd4e3f3151eb3ee2d7e3b46d9de6c'
        },
        'realAWSCloudwatch': {
            'ec2_cpu_utilization_24ae8d.csv': '670f5fea6e97b1725fdfc1d7ea8c5249',
            'ec2_cpu_utilization_53ea38.csv': '2ccb238a824fe1a0d5ba40edd8bd7258',
            'ec2_cpu_utilization_5f5533.csv': '852c8473584b2bb4b34a2cc107b55eda',
            'ec2_cpu_utilization_77c1ca.csv': 'e9259ab08445b0584eaf7169f744c4b0',
            'ec2_cpu_utilization_825cc2.csv': '85ed1cc00d26de5c921a0af2c6019a5b',
            'ec2_cpu_utilization_ac20cd.csv': '14a0e0d6b9d46a2dbdcfac9464336c02',
            'ec2_cpu_utilization_c6585a.csv': 'fb19d87d3e38a725da145681a86ff442',
            'ec2_cpu_utilization_fe7f93.csv': '5211650f0dfd20bf8f1304ba46c37e0f',
            'ec2_disk_write_bytes_1ef3de.csv': 'be2a8194a63fee9db0a0fff5a13595ae',
            'ec2_disk_write_bytes_c0d644.csv': 'c96f95d28c564d050179954973d7c3c2',
            'ec2_network_in_257a54.csv': '82ac6c4d6279d402465f18ea5a182ec1',
            'ec2_network_in_5abac7.csv': 'd7c530dccbef98bc4c9cabee966e5eea',
            'elb_request_count_8c0756.csv': 'f8cadcb1b5622b7087b86a6f299b5b7c',
            'grok_asg_anomaly.csv': 'bc88304f35513e99d592bdec3663104c',
            'iio_us-east-1_i-a2eb1cd9_NetworkIn.csv': '06483b7cc2be2078ca4bc9c88192a828',
            'rds_cpu_utilization_cc0c53.csv': 'e1498abcc54c6564e81d26c47c837365',
            'rds_cpu_utilization_e47b3b.csv': 'e2a7925707f7296a884190070d43ee0d'
        },
        'realKnownCause': {
            'ambient_temperature_system_failure.csv': '985aecd214508bf394810d74406e983b',
            'cpu_utilization_asg_misconfiguration.csv': '093b42c79df6c41a4ba25d95e72cc982',
            'ec2_request_latency_system_failure.csv': '45af0ecfa6784420b02fa7e504c25c8c',
            'machine_temperature_system_failure.csv': '5ba758686356bda5f63e7b10093394c7',
            'nyc_taxi.csv': '0c71fc23265dfa34ce7ff6c8459cd018',
            'rogue_agent_key_hold.csv': '4e359a0dc93343851951fb6ef54a5172',
            'rogue_agent_key_updown.csv': '871e4b50df869a319e62d11d11a52b46'
        },
        'realTraffic': {
            'occupancy_6005.csv': '3c93a137d33b00fc15de5cd16c83628f',
            'occupancy_t4013.csv': '0d23e04a73b439cb3394c4d132ec2697',
            'speed_6005.csv': '350969d1826a25aa8fc606398f0d66e0',
            'speed_7578.csv': '3fe21df810be6b31520d83b6f889d8b5',
            'speed_t4013.csv': '385199e156951a63d87c7d839eee2224',
            'TravelTime_387.csv': 'd0af3020e4e15cbe840a2cafbb5750df',
            'TravelTime_451.csv': 'cef6762eb4b297399e04fe16a325fa78'
        },
        'realTweets': {
            'Twitter_volume_AAPL.csv': '7bddd2fc590467dddec779bd0609a5e6',
            'Twitter_volume_AMZN.csv': '67d02ba74a5edea23042538536babf67',
            'Twitter_volume_CRM.csv': 'c994d7725b16fca36bbf3497400ba631',
            'Twitter_volume_CVS.csv': '6c4cc222fd3b6d948d1651d233c8bdad',
            'Twitter_volume_FB.csv': '2bc29f4882cb2bf877f05ea51310d789',
            'Twitter_volume_GOOG.csv': '85cfdc9d32048f75b45d92c91672783e',
            'Twitter_volume_IBM.csv': '7c0e103323865c0d19b5ac873bdc9a3d',
            'Twitter_volume_KO.csv': '9b4be473a282678bbf7f1012798bc587',
            'Twitter_volume_PFE.csv': '6b377884b5c11aeb56b7e98dc7e7fd55',
            'Twitter_volume_UPS.csv': 'ca399cae5c4d94d303452f38710e62cd'
        }
    }

    def __init__(self, root: str = None, subset: str = None, download: bool = True):
        assert subset in self.__subsets, f'Available subsets: {self.__subsets}'

        if root is None:
            root_path = Path.home() / 'nab'
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
                print('Start decompressing...')
                decompress_file(root_path / self.__filename, root_path)
                for folder_name in self.__file_list:
                    shutil.move(str(root_path / 'NAB-1.1' / 'data' / folder_name), str(root_path))
                shutil.move(str(root_path / 'NAB-1.1' / 'labels' / self.__label_filename), str(root_path))
                shutil.rmtree(str(root_path / 'NAB-1.1'))
                assert self.__check_integrity(root_path)
        else:
            assert self.__check_integrity(root_path)

        data = []
        labels = []
        timestamps = []

        with (root_path / self.__label_filename).open('r') as f:
            label_dict = json.load(f)
        for csv_file in (root_path / subset).glob('*.csv'):
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            label = np.zeros(len(df))
            for anomaly_window in label_dict[subset + '/' + csv_file.name]:
                st, ed = pd.to_datetime(anomaly_window)
                label[(df['timestamp'] >= st).values & (df['timestamp'] <= ed).values] = 1
            df['label'] = label

            data.append(df['value'].values.reshape(-1, 1))
            labels.append(df['label'].values.astype(np.long).reshape(-1))
            timestamps.append(df['timestamp'].values.reshape(-1))

        super(NABDataset, self).__init__(data_list=data, label_list=labels, timestamp_list=timestamps)

    def __check_integrity(self, root: Union[str, Path]):
        if isinstance(root, str):
            root = Path(root)

        for folder_name, folder_dict in self.__file_list.items():
            for file_name, file_md5 in folder_dict.items():
                if not check_existence(root / folder_name / file_name, file_md5):
                    return False

        if not check_existence(root / self.__label_filename, self.__label_md5):
            return False

        return True
