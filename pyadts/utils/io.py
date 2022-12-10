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
import zipfile
from pathlib import Path
from typing import Union, Dict, IO
from urllib.request import urlopen

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


def download_link(link: str, dest_path: Union[str, Path], make_dir: bool = True, chunk_size: int = 1024,
                  verbose: bool = True):
    if isinstance(dest_path, str):
        dest_path = Path(dest_path)
    folder = dest_path.parent

    if make_dir and (not folder.exists()):
        folder.mkdir()

    if verbose:
        print(f'[INFO] downloading from `{link}`...')
    with urlopen(link) as response:
        with dest_path.open('wb') as f:
            for chunk in tqdm(iter(lambda: response.read(chunk_size), ""), total=response.length):
                if not chunk:
                    break
                f.write(chunk)


def calculate_md5(file_path: Union[str, Path], chunk_size: int = 1024 * 1024):
    md5 = hashlib.md5()
    if isinstance(file_path, str):
        file_path = Path(file_path)

    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)

    return md5.hexdigest()


def check_md5(file_path: Union[str, Path], md5: str):
    return md5 == calculate_md5(file_path)


def check_existence(file_path: Union[str, Path], md5: str = None):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.is_file():
        return False
    if md5 is None:
        return True
    return check_md5(file_path, md5)


def __extract_zip(file_path: str, dest_path: str):
    with zipfile.ZipFile(file_path, 'r', zipfile.ZIP_STORED) as zip:
        zip.extractall(dest_path)


def __extract_tar(file_path: str, dest_path: str):
    with tarfile.open(file_path, 'r') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, dest_path)


#
# def __extract_gz(file_path: str, dest_path: str):
#     with gzip.open(file_path, 'rb') as gz:
#         with open(dest_path, 'wb') as f:
#             f.write(gz.read())


def decompress_file(file_path: Union[str, Path], dest_path: Union[str, Path]):
    if isinstance(file_path, Path):
        file_path = str(file_path)
    if isinstance(dest_path, Path):
        dest_path = str(dest_path)

    assert os.path.isfile(file_path)

    if file_path.endswith('.tar') or file_path.endswith('.tar.gz'):
        __extract_tar(file_path, dest_path)
    # elif file_path.endswith('.gz'):
    #     __extract_gz(file_path, dest_path)
    elif file_path.endswith('.zip'):
        __extract_zip(file_path, dest_path)
    else:
        raise ValueError('Unsupported file type!')
