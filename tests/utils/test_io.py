"""
@Time    : 2021/11/9 12:17
@File    : test_io.py
@Software: PyCharm
@Desc    : 
"""
import os

from pyadts.utils.io import download_link, decompress_file, check_existence


def test_download_link():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    download_link(url, 'tmp/train-images-idx3-ubyte.gz')
    assert check_existence('tmp/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873')


def test_decompress_file():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    url = 'https://bitbucket.org/gsudmlab/mvtsdata_toolkit/downloads/petdataset_01.zip'
    download_link(url, 'tmp/petdataset_01.zip')
    decompress_file('tmp/petdataset_01.zip', 'tmp')
