# Anomaly Detection for Time-series (PyADT)

`PyADT` is aimed at accelerating the workflow of time series anomaly detection for researchers. It contains various utilities for data loading, pre-processing, detector construction, detector ensemble, evaluation and etc. `PyADT` can help you to write less boilerplate on following parts:

- Preparing dataset & pre-processing
- Feature extraction (Optional)
- Model training
- Ensemble (Optional)
- Evaluation

## Installation
To install the package locally, run:

```bash
cd <pyadt_dir>
pip install .
```

## Quick Start

### Fetch the dataset



### Pre-processing



### Feature extraction



### Train the model



### Ensemble



### Evaluation



### The pipeline



## Other Utilities

### Visualization



## Implemented Algorithms

### Supervised Approaches

- Random Forest[^1]

- SVM[^2]
- XGBoost[^3]
- LightGBM[^4]
- Deep Neural Network[^5]

### Unsupervised Approaches

#### Non-parametric

- SR[^6]
- Threshold

#### Statistic-based

- SPOT
- DSPOT

#### Machine learning-based

- LOF
- Isolation Forest
- OCSVM

#### Deep-based

- Autoencoder
- Donut

[^1]: Breiman L. Random forests[J]. Machine learning, 2001, 45(1): 5-32.
[^2]: Cortes C, Vapnik V. Support-vector networks[J]. Machine learning, 1995, 20(3): 273-297.
[^3]: Chen T, He T, Benesty M, et al. Xgboost: extreme gradient boosting[J]. R package version 0.4-2, 2015: 1-4.
[^4]: Ke G, Meng Q, Finley T, et al. Lightgbm: A highly efficient gradient boosting decision tree[C]. Advances in neural information processing systems. 2017: 3146-3154.
[^5]: Goodfellow I, Bengio Y, Courville A, et al. Deep learning[M]. Cambridge: MIT press, 2016.
[^6]: Ren H, Xu B, Wang Y, et al. Time-Series Anomaly Detection Service at Microsoft[C]. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019: 3009-3017.