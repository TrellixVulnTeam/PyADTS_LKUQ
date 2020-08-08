from setuptools import setup


def parse_requirements(file_name):
    with open(file_name, 'r') as f:
        line_striped = (line.strip() for line in f.readlines())
    return [line for line in line_striped if line and not line.startswith('#')]


setup(
    name='pyadt',
    author='Xiao Qinfeng',
    description='A python package for time series anomaly detection',
    long_description=open('README.md').read(),
    version='0.1',
    packages=[],
    scripts=[],
    install_requirements=parse_requirements('requirements.txt'),
    url='https://github.com/larryshaw0079/PyADT',
    license='GPL-3.0 License'
)
