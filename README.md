# Anomaly Detection for Time-series (PyADT)

This project contains multiple KPI anomaly detection algorithms.

[TOC]

## Basic Usage

### Fetch dataset

```python
from torchts.dataset.repository import get_kpi_dataset, KPIDataset

kpi_series = get_kpi_dataset(index=0, download=True, method='wget')
kpi_dataset = KPIDataset(kpi_series=kpi_series, window_size=100)

```

### Visualization

```python
from torchts.dataset.repository import get_kpi_dataset, KPIDataset

kpi_series = get_kpi_dataset(index=0, download=True, method='wget')
kpi_series.plot()
```

### Evaluation

```python
import numpy as np

from torchts.evaluation.metrics import best_f1_with_delay, pr_auc_with_delay, roc_auc_with_delay

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
