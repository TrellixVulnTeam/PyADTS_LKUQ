import os
from typing import Tuple

import pandas as pd

from ..utils import timestamp_to_datetime

DATA_NAMES = {
    'first': 'phase2_train.csv',
    'second': 'phase2_ground_truth.hdf'
}

KPI_IDS = ['05f10d3a-239c-3bef-9bdc-a2feeb0037aa',
           '0efb375b-b902-3661-ab23-9a0bb799f4e3',
           '1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0',
           '301c70d8-1630-35ac-8f96-bc1b6f4359ea',
           '42d6616d-c9c5-370a-a8ba-17ead74f3114',
           '43115f2a-baeb-3b01-96f7-4ea14188343c',
           '431a8542-c468-3988-a508-3afd06a218da',
           '4d2af31a-9916-3d9f-8a8e-8a268a48c095',
           '54350a12-7a9d-3ca8-b81f-f886b9d156fd',
           '55f8b8b8-b659-38df-b3df-e4a5a8a54bc9',
           '57051487-3a40-3828-9084-a12f7f23ee38',
           '6a757df4-95e5-3357-8406-165e2bd49360',
           '6d1114ae-be04-3c46-b5aa-be1a003a57cd',
           '6efa3a07-4544-34a0-b921-a155bd1a05e8',
           '7103fa0f-cac4-314f-addc-866190247439',
           '847e8ecc-f8d2-3a93-9107-f367a0aab37d',
           '8723f0fb-eaef-32e6-b372-6034c9c04b80',
           '9c639a46-34c8-39bc-aaf0-9144b37adfc8',
           'a07ac296-de40-3a7c-8df3-91f642cc14d0',
           'a8c06b47-cc41-3738-9110-12df0ee4c721',
           'ab216663-dcc2-3a24-b1ee-2c3e550e06c9',
           'adb2fde9-8589-3f5b-a410-5fe14386c7af',
           'ba5f3328-9f3f-3ff5-a683-84437d16d554',
           'c02607e8-7399-3dde-9d28-8a8da5e5d251',
           'c69a50cf-ee03-3bd7-831e-407d36c7ee91',
           'da10a69f-d836-3baa-ad40-3e548ecf1fbd',
           'e0747cad-8dc8-38a9-a9ab-855b61f5551d',
           'f0932edd-6400-3e63-9559-0a9860a1baa9',
           'ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa']


def __check_dataset(root_path: str):
    if (not os.path.exists(os.path.join(root_path, DATA_NAMES['first']))) or (
            not os.path.exists(os.path.join(root_path, DATA_NAMES['second']))):
        raise FileNotFoundError('The dataset is not found in path %s' % root_path)


def get_kpi(root_path: str, kpi_id: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Parameters
    ----------
    root_path : str
        The path of downloaded kpi dataset.
    kpi_id : int
        The index of selected kpi series (The whole dataset contains multiple series).

    Returns
    -------
    pd.DataFrame
        The data DataFrame
    pd.DataFrame
        The meta DataFrame
    """
    __check_dataset(root_path)

    first_df = pd.read_csv(os.path.join(root_path, DATA_NAMES['first']))
    second_df = pd.read_hdf(os.path.join(root_path, DATA_NAMES['second']))
    df = pd.concat([first_df, second_df])

    selected_df = df[df['KPI ID'].apply(str) == KPI_IDS[kpi_id]]

    value = selected_df['value'].values
    label = selected_df['label'].values
    timestamp = selected_df['timestamp'].values
    datetime = pd.to_datetime(selected_df['timestamp'].apply(timestamp_to_datetime))

    data_df = pd.DataFrame({'value': value}, index=datetime)
    meta_df = pd.DataFrame({'label': label, 'timestamp': timestamp}, index=datetime)

    return data_df, meta_df
