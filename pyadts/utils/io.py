"""
@Time    : 2021/10/25 11:06
@File    : io.py
@Software: PyCharm
@Desc    : 
"""
import pickle
from typing import Union, Dict, IO


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
