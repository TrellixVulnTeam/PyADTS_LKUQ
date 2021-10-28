"""
@Time    : 2021/10/25 11:06
@File    : io.py
@Software: PyCharm
@Desc    : 
"""
import hashlib
import os
import pickle
import tarfile
import urllib
import zipfile
from typing import Union, Dict, IO

from tqdm.std import tqdm


def save_objects(objs: Dict, f: Union[str, IO]):
    if isinstance(f, str):
        f = open(f, 'wb')

    pickle.dump(objs, f)
    f.close()


def load_objects(f: Union[str, IO]):
    if isinstance(f, str):
        f = open(f, 'rb')

    objs = pickle.load(f)
    f.close()

    return objs


def download_link(link: str, dest_path: str, make_dir: bool = True, chunk_size: int = 1024):
    folder = os.path.split(dest_path)[0]

    if make_dir and (not os.path.exists(folder)):
        os.makedirs(folder)

    with urllib.request.urlopen(link) as response:
        with open(dest_path, 'wb') as f:
            for chunk in tqdm(iter(lambda: response.read(chunk_size), ""), total=response.length):
                if not chunk:
                    break
                f.write(chunk)


def calculate_md5(file_path: str, chunk_size: int = 1024 * 1024):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)

    return md5.hexdigest()


def check_md5(file_path: str, md5: str):
    return md5 == calculate_md5(file_path)


def check_existence(file_path: str, md5: str = None):
    if not os.path.isfile(file_path):
        return False
    if md5 is None:
        return True
    return check_md5(file_path, md5)


def __extract_zip(file_path: str, dest_path: str):
    with zipfile.ZipFile(file_path, 'r', zipfile.ZIP_STORED) as zip:
        zip.extractall(dest_path)


def __extract_tar(file_path: str, dest_path: str):
    with tarfile.open(file_path, 'r') as tar:
        tar.extractall(dest_path)


def decompress_file(file_path: str, dest_path: str):
    assert os.path.isfile(file_path)

    if file_path.endswith('.tar') or file_path.endswith('.tar.gz'):
        __extract_tar(file_path, dest_path)
    elif file_path.endswith('.zip'):
        __extract_zip(file_path, dest_path)
    else:
        raise ValueError('Unsupported file type!')
