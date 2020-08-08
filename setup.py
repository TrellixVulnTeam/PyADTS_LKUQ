import os
import atexit
import glob
import shutil

import matplotlib
from setuptools import setup
from setuptools.command.install import install


# Get description from README
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()


def parse_requirements(file_name):
    with open(file_name, 'r') as f:
        line_striped = (line.strip() for line in f.readlines())
    return [line for line in line_striped if line and not line.startswith('#')]


def install_styles():
    # Find all style files
    stylefiles = glob.glob('style/*.mplstyle', recursive=True)

    # Find stylelib directory (where the *.mplstyle files go)
    mpl_stylelib_dir = os.path.join(matplotlib.get_configdir(), "stylelib")
    if not os.path.exists(mpl_stylelib_dir):
        os.makedirs(mpl_stylelib_dir)

    # Copy files over
    print("Installing styles into", mpl_stylelib_dir)
    for stylefile in stylefiles:
        print(os.path.basename(stylefile))
        shutil.copy(
            stylefile,
            os.path.join(mpl_stylelib_dir, os.path.basename(stylefile)))


class PostInstallMoveFile(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(install_styles)


setup(
    name='pyadt',
    author='Xiao Qinfeng',
    description='A python package for time series anomaly detection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.1',
    packages=[],
    scripts=[],
    install_requirements=parse_requirements('requirements.txt'),
    cmdclass={'install': PostInstallMoveFile},
    url='https://github.com/larryshaw0079/PyADT',
    license='GPL-3.0 License'
)
