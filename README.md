# Anomaly Detection for Time-series (PyADTS)

[![Build Status](https://travis-ci.org/larryshaw0079/PyADTS.svg?branch=master)](https://travis-ci.org/larryshaw0079/PyADTS)
[![Coverage Status](https://coveralls.io/repos/github/larryshaw0079/PyADTS/badge.svg?branch=master)](https://coveralls.io/github/larryshaw0079/PyADTS?branch=master)
[![Documentation Status](https://readthedocs.org/projects/pyadts/badge/?version=latest)](https://pyadts.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/pyadts)](https://pypi.org/project/pyadts/)
[![Anaconda](https://anaconda.org/larryshaw0079/pyadts/badges/version.svg)](https://anaconda.org/larryshaw0079/pyadts)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/larryshaw0079/PyADTS/master?filepath=notebooks%2Fdemo.ipynb)
---

`PyADTS` is aimed at **accelerating the workflow of time series anomaly detection for researchers**. It contains various utilities for data loading, pre-processing, detector construction, detector ensemble, evaluation and etc. `PyADTS` can help you to write less boilerplate on following parts:

- Preparing dataset & pre-processing
- Feature extraction (Optional)
- Model training
- Ensemble (Optional)
- Evaluation

With `PyADTS`, you can run baselines rapidly.

You can find the complete [documentation](https://pyadts.readthedocs.io/en/latest/) here.

**Table of Contents**:

- [Anomaly Detection for Time-series (PyADTS)](#anomaly-detection-for-time-series--pyadt-)
  * [Installation](#installation)
  * [Quick Start](#quick-start)
  * [Other Utilities](#other-utilities)
  * [Implemented Algorithms](#implemented-algorithms)

## Installation
To install the package locally, run:

```bash
>> cd <pyadts_dir>
>> pip install .
```

To install the package from PyPi, run:

```bash
>> pip install pyadts
```

## Quick Start

### Fetch the dataset

`PyADTS` contains various built-in datasets. To utilize them:

```python
from pyadts.data import get_nab_nyc_taxi

data = get_nab_nyc_taxi(root_path='<the_path_of_nab_dataset>')
```

All components of the dataset are organized as a dict:

`{'value': value, 'label': label, 'timestamp': timestamp, 'datetime': datetime}`

### Pre-processing

It's important to pre-process the time series before training. `PyADTS` offered three types of pre-processing methods including:

- Rearrangement: Sort the values along with the timestamp and reconstruct the timestamp to discover missing values. (return a dict and append an attribute `missing`)
- Normalization: Normalize the time series
- Imputation: Impute the time series.

```python
from pyadts.data import series_impute, series_normalize, series_rearrange

data_processed = series_rearrange(**data)

data_processed['value'] = series_normalize(data_processed['value'], mask=data_processed['missing'], method='zscore')

data_processed['value'] = series_impute(data_processed['value'], missing=data_processed['missing'], method='linear')
```

### Feature extraction

Extracting manual features is essential for some anomaly detection approaches. `PyADT` offered various options for extracting features including:

- Simple features: logarithm, difference, second-order difference, ...
- Window-based features: window mean value, window std value, ...
- Decomposition features: STL decomposition, ...
- Frequency domain features: wavelet features, spectral residual, ...
- Regression features: SARIMA regression residual, Exponential Smoothing residual, ...

```python
from pyadts.data import FeatureExtractor

feature_extractor = FeatureExtractor()
```

### Train the model

Different anomaly detection algorithms should be utilized to tackle different scenarios. `PyADT` contains various algorithms including supervised-, unsupervised-, nonparametric-methods (you can refer the full list of [implemented algorithms](#implemented-algorithms)).

```python
from pyadts import ThresholdDetector

train_x = data['value']
detector = ThresholdDetector()
pred_y = detector.fit_predict(train_x)
```

### Ensemble

TODO

### Evaluation

It's easy to evaluate your algorithms using `PyADT`'s built-in metrics:

```python
from pyadts import roc_auc

train_y = data['label']
roc = roc_auc(pred_y, train_y, delay=7)
```

In real-world applications, the delay of anomaly alerts is acceptable. So `PyADT` offered the `delay` argument for all metrics.

<img src="https://i.loli.net/2020/08/12/shGMx2QqjcP8tTe.png" style="zoom: 50%;" />

### The pipeline

TODO

## Other Utilities

### Visualization

You can visualize your data with a single line of code:

```python
from pyadts.data import plot_series

fig = plot_series(value=data['value'], label=data['label'], datetime=data['datetime'], plot_vline=True)
fig.show()
```

The example visualization:

<img src="https://i.loli.net/2020/08/12/j78NoQsZHtR5lnv.png" style="zoom: 50%;" />

### Generate synthetic data

TODO

## Implemented Algorithms

### Supervised Approaches

- Random Forest
- SVM
- Deep Neural Network

### Unsupervised Approaches

#### Non-parametric

- SR
- Threshold
- Quantile
- K​-Sigma

#### Statistic-based

- SPOT
- DSPOT
- Autoregression
- ESD
- S-ESD
- S-H-ESD

#### Machine learning-based

- LOF
- Isolation Forest
- OCSVM

#### Deep-based

- Autoencoder
- RNN Autoencoder
- Donut

## TODO

- [ ] Fix bugs and reformat code
- [ ] Multi-variate time series support
- [ ] Complete the models
- [ ] Incorporating Numba
- [ ] Implement the pipeline
- [ ] Synthetic data generator
- [ ] Allow both inplace and non-inplace operations
- [ ] Add streaming support
