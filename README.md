# Anomaly Detection for Time-series (PyADT)

This project contains multiple KPI anomaly detection algorithms.

- [Anomaly Detection for Time-series (PyADT)](#anomaly-detection-for-time-series--pyadt-)
  * [Basic Usage](#basic-usage)
    + [Fetch dataset](#fetch-dataset)
    + [Visualization](#visualization)
    + [Evaluation](#evaluation)
  * [Implemented Algorithms](#implemented-algorithms)
    + [Supervised Algorithms](#supervised-algorithms)
    + [Unsupervised Algorithms](#unsupervised-algorithms)
      - [Machine Learning Models](#machine-learning-models)
      - [Deep Models](#deep-models)

## Installation
To install the package locally, run:

```bash
cd <pyadt_dir>
pip install .
```

## User Guide

### Fetch dataset

```python
from pyadt.datasets.series import Series
from pyadt.datasets.repository.nab import get_nab_nyc_taxi

series = get_nab_nyc_taxi('./data/nab')
```

### Visualization

```python
from pyadt.utils.visualization import plot

fig = plot(series.feature, series.label, series.timestamp)
fig.show()
```

### Evaluation

```python
import numpy as np

from pyadt.metrics import best_f1_with_delay, pr_auc_with_delay, roc_auc_with_delay

# Pseudo results
y_true = np.random.randint(low=0, high=1, size=100)
y_score = np.clip(np.abs(np.random.randn(y_true.shape[0]).astype(np.float32)), a_min=0.0, a_max=1.0)

f1 = best_f1_with_delay(y_score, y_true, delay=7)
pr_auc = pr_auc_with_delay(y_score, y_true, delay=7)
roc_auc = roc_auc_with_delay(y_score, y_true, delay=7)

print(f1, pr_auc, roc_auc)
```

## Implemented Algorithms

### Supervised Algorithms

- Random Forest
- XGBoost
- LightGBM
- Deep Neural Network

### Unsupervised Algorithms

#### Machine Learning Models

- LOF
- Isolation Forest
- SR
- SPOT
- DSPOT

#### Deep Models

- Autoencoder
- Donut
