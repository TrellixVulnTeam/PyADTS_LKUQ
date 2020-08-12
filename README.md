# Anomaly Detection for Time-series (PyADT)

`PyADT` is aimed at accelerating the workflow of time series anomaly detection for researchers. It contains various utilities for data loading, pre-processing, detector construction, detector ensemble, evaluation and etc. `PyADT` can help you to write less boilerplate on following parts:

- Preparing dataset & pre-processing
- Feature extraction (Optional)
- Model training
- Ensemble (Optional)
- Evaluation

- [Anomaly Detection for Time-series (PyADT)](#anomaly-detection-for-time-series--pyadt-)
  * [Installation](#installation)
  * [Quick Start](#quick-start)
    + [Fetch the dataset](#fetch-the-dataset)
    + [Pre-processing](#pre-processing)
    + [Feature extraction](#feature-extraction)
    + [Train the model](#train-the-model)
    + [Ensemble](#ensemble)
    + [Evaluation](#evaluation)
    + [The pipeline](#the-pipeline)
  * [Other Utilities](#other-utilities)
    + [Visualization](#visualization)
  * [Implemented Algorithms](#implemented-algorithms)
    + [Supervised Approaches](#supervised-approaches)
    + [Unsupervised Approaches](#unsupervised-approaches)
      - [Non-parametric](#non-parametric)
      - [Statistic-based](#statistic-based)
      - [Machine learning-based](#machine-learning-based)
      - [Deep-based](#deep-based)

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

- Random Forest
- SVM
- XGBoost
- LightGBM
- Deep Neural Network

### Unsupervised Approaches

#### Non-parametric

- SR
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
