"""
@Time    : 2021/10/28 2:02
@File    : creditcard.py
@Software: PyCharm
@Desc    :
"""
import os

import numpy as np
import pandas as pd
from tqdm.std import tqdm

from pyadts.generic import TimeSeriesRepository
from pyadts.preprocessing import rearrange, impute


# KPI_IDS = ['05f10d3a-239c-3bef-9bdc-a2feeb0037aa',
#            '0efb375b-b902-3661-ab23-9a0bb799f4e3',
#            '1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0',
#            '301c70d8-1630-35ac-8f96-bc1b6f4359ea',
#            '42d6616d-c9c5-370a-a8ba-17ead74f3114',
#            '43115f2a-baeb-3b01-96f7-4ea14188343c',
#            '431a8542-c468-3988-a508-3afd06a218da',
#            '4d2af31a-9916-3d9f-8a8e-8a268a48c095',
#            '54350a12-7a9d-3ca8-b81f-f886b9d156fd',
#            '55f8b8b8-b659-38df-b3df-e4a5a8a54bc9',
#            '57051487-3a40-3828-9084-a12f7f23ee38',
#            '6a757df4-95e5-3357-8406-165e2bd49360',
#            '6d1114ae-be04-3c46-b5aa-be1a003a57cd',
#            '6efa3a07-4544-34a0-b921-a155bd1a05e8',
#            '7103fa0f-cac4-314f-addc-866190247439',
#            '847e8ecc-f8d2-3a93-9107-f367a0aab37d',
#            '8723f0fb-eaef-32e6-b372-6034c9c04b80',
#            '9c639a46-34c8-39bc-aaf0-9144b37adfc8',
#            'a07ac296-de40-3a7c-8df3-91f642cc14d0',
#            'a8c06b47-cc41-3738-9110-12df0ee4c721',
#            'ab216663-dcc2-3a24-b1ee-2c3e550e06c9',
#            'adb2fde9-8589-3f5b-a410-5fe14386c7af',
#            'ba5f3328-9f3f-3ff5-a683-84437d16d554',
#            'c02607e8-7399-3dde-9d28-8a8da5e5d251',
#            'c69a50cf-ee03-3bd7-831e-407d36c7ee91',
#            'da10a69f-d836-3baa-ad40-3e548ecf1fbd',
#            'e0747cad-8dc8-38a9-a9ab-855b61f5551d',
#            'f0932edd-6400-3e63-9559-0a9860a1baa9',
#            'ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa']


# def get_kpi_dataset(root: str, subset: Iterable[int] = None, transform: Transform = None, download: bool = False,
#                     impute: bool = True, normalize: bool = True,
#                     rearrange: bool = True, split_method: str = 'constant', splits: List[Union[int, float]] = None,
#                     kfolds: int = None):
#     if download:
#         raise ValueError('The KPI dataset should be downloaded manually. '
#                          'Please download the dataset at `http://iops.ai/dataset_detail/?id=7`!')
#
#     assert split_method in ['constant', 'kfold', 'leave_one_out']
#
#     first_df = pd.read_csv(os.path.join(root, SPLITS['first']))
#     second_df = pd.read_hdf(os.path.join(root, SPLITS['second']))
#     df = pd.concat([first_df, second_df])
#
#     if split_method == 'constant':
#         pass
#     elif split_method == 'kfold':
#         assert kfolds is not None, 'The number of folds must be specified!'
#
#         kf = KFold(n_splits=kfolds)
#     elif split_method == 'leave_one_out':
#         loo = LeaveOneOut()
#
#
#     else:
#         raise ValueError

# selected_df = df[df['KPI ID'].apply(str) == KPI_IDS[kpi_id]]
#
# value = selected_df['value'].values
# label = selected_df['label'].values
# timestamp = selected_df['timestamp'].values
# datetime = pd.to_datetime(selected_df['timestamp'].apply(timestamp_to_datetime))
#
# data_df = pd.DataFrame({'value': value}, index=datetime)
# meta_df = pd.DataFrame({'label': label, 'timestamp': timestamp}, index=datetime)


class KPIDataset(TimeSeriesRepository):
    __splits = {
        'first': 'phase2_train.csv',
        'second': 'phase2_ground_truth.hdf'
    }

    def __init__(self, root: str, download: bool = False):
        super(KPIDataset, self).__init__()

        if download:
            raise ValueError('The KPI dataset should be downloaded manually. '
                             'Please download the dataset at `http://iops.ai/dataset_detail/?id=7`!')

        first_df = pd.read_csv(os.path.join(root, self.__splits['first']))
        second_df = pd.read_hdf(os.path.join(root, self.__splits['second']))
        df = pd.concat([first_df, second_df])

        kpi_ids = np.unique(df['KPI ID'].values.astype(str))
        df_group_by_id = {kpi: df[df['KPI ID'] == kpi] for kpi in kpi_ids}

        self.data = []
        self.labels = []

        for key, df in tqdm(df_group_by_id.items(), desc='::LOADING DATA::'):
            value = df['value'].values
            timestamp = df['timestamp'].values
            label = df['label'].values

            rearrange(value, timestamp, label)
            impute(value)

            self.data.append(value.reshape(1, -1))
            self.labels.append(label.reshape(-1))

        self.sep_indicators = np.cumsum([item.shape[-1] for item in self.data])
        self.data = np.concatenate(self.data, axis=-1)
        self.labels = np.concatenate(self.labels)
