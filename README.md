# PyADTS: A Python Toolkit for Time-series Anomaly Detection

<div align=center><img src="https://i.loli.net/2020/09/16/jlALoQ18vnXDPCU.png" width=45% height=45% /></div >

`PyADTS` is aimed at **accelerating the workflow of time series anomaly detection for researchers**. It contains various
modules for data loading, pre-processing, anomaly detection, detector ensembling, evaluation and etc. `PyADTS` can help
you to write less boilerplate on following parts:

- Preparing dataset & pre-processing
- Anomaly detection
- Detector ensembling
- Anomaly score calibration
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
>> git clone https://github.com/larryshaw0079/PyADTS
>> cd PyADTS
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

### Simple Detectors

| Algo              | Desc | Title | Year | Ref  |
| ----------------- | ---- | ----- | ---- | ---- |
| Range Detector    |      | -     | -    | -    |
| Quantile Detector |      | -     | -    | -    |
| Gradient Detector |      | -     | -    | -    |

### Statistical Approaches

| Algo               | Desc | Title | Year | Ref          |
| ------------------ | ---- | ----- | ---- | ------------ |
| KSigma             |      | -     | -    | -            |
| InterQuartileRange |      | -     | -    | -            |
| Hotelling          |      |       |      | [[1]](#ref1) |
| ESD                |      |       |      | [[2]](#ref2) |
| Matrix Profile     |      |       |      | [[3]](#ref3) |
| SPOT               |      |       |      | [[4]](#ref4) |
| DSPOT              |      |       |      | [[4]](#ref4) |
| SR                 |      |       |      | [[5]](#ref5) |

### Machine Learning Approaches

| Algo             | Desc | Title | Year | Ref          |
| ---------------- | ---- | ----- | ---- | ------------ |
| Autoregression   |      | -     | -    | -            |
| RRCF             |      |       |      | [[6]](#ref6) |
| Isolation Forest |      |       |      | [[7]](#ref7) |

### Deep Approaches

| Algo        | Desc | Title | Year | Ref            |
| ----------- | ---- | ----- | ---- | -------------- |
| Autoencoder |      | -     | -    | -              |
| VAE         |      |       |      | [[8]](#ref8)   |
| Donut       |      |       |      | [[9]](#ref9)   |
| MSCRED      |      |       |      | [[10]](#ref10) |
| OmniAnomaly |      |       |      | [[11]](#ref11) |
| USAD        |      |       |      | [[12]](#ref12) |

## Datasets

### Univariate

- [**KPI**](https://github.com/NetManAIOps/KPI-Anomaly-Detection)
- [**NAB**](https://github.com/numenta/NAB) [[13]](#ref13)
- [**Yahoo**](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70)

### Multivariate

- [**SMAP**](https://github.com/khundman/telemanom) [[14]](#ref14)
- [**MSL**](https://github.com/khundman/telemanom) [[14]](#ref14)
- [**SMD**](https://github.com/NetManAIOps/OmniAnomaly) [[11]](#ref11)
- [**SKAB**](https://github.com/waico/SkAB) [[15]](#ref15)
- [**SWaT**](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) [[16]](#ref16)
- [**CIC-IDS**](https://www.unb.ca/cic/datasets/ids-2017.html) [[17]](#ref17)
- [**Credit Card**](https://www.openml.org/d/1597) [[18]](#ref18)
- [**GECCO**](https://zenodo.org/record/3884398#.YX9lOJpBzEY) [[19]](#ref19)
- [**SWANSF**](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM) [[20]](#ref20)

## Reference

> <div id="ref1">
>[1] Hotelling, H. (1947). Multivariate Quality Control-illustrated by the air testing of sample bombsights.
></div>
>
><div id="ref2">
>[2] Rosner, B. (1983). Percentage Points for a Generalized ESD Many-Outlier Procedure. Technometrics, 25(2), 165–172. https://doi.org/10.2307/1268549
></div>
>
><div id="ref3">
>[3] C. M. Yeh et al. Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View That Includes Motifs, Discords and Shapelets. 2016 IEEE 16th International Conference on Data Mining (ICDM), 2016, pp. 1317-1322, doi: 10.1109/ICDM.2016.0179.
></div>
>
><div id="ref4">
>[4] Alban Siffer, Pierre-Alain Fouque, Alexandre Termier, and Christine Largouet. 2017. Anomaly Detection in Streams with Extreme Value Theory. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '17). Association for Computing Machinery, New York, NY, USA, 1067–1075. DOI:https://doi.org/10.1145/3097983.3098144
></div>
>
><div id="ref5">
>[5] Hansheng Ren, Bixiong Xu, Yujing Wang, Chao Yi, Congrui Huang, Xiaoyu Kou, Tony Xing, Mao Yang, Jie Tong, and Qi Zhang. 2019. Time-Series Anomaly Detection Service at Microsoft. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19). Association for Computing Machinery, New York, NY, USA, 3009–3017.
></div>
>
><div id="ref6">
>[6] S. Guha, N. Mishra, G. Roy, & O. Schrijvers, Robust random cut forest based anomaly detection on streams, in Proceedings of the 33rd International conference on machine learning, New York, NY, 2016 (pp. 2712-2721).
></div>
>
><div id="ref7">
>[7] F. T. Liu, K. M. Ting and Z. Zhou. Isolation Forest. 2008 Eighth IEEE International Conference on Data Mining, 2008, pp. 413-422, doi: 10.1109/ICDM.2008.17.
></div>
>
><div id="ref8">
>[8] Kingma, D.P., Welling, M. (2014). Auto-Encoding Variational Bayes. 2nd International Conference on Learning Representations, ICLR 2014.
></div>
>
><div id="ref9">
>[9] Xu, H., Chen, W., Zhao, N., Li, Z., Bu, J., Li, Z., Liu, Y., Zhao, Y., Pei, D., Feng, Y., Chen, J.J., Wang, Z., & Qiao, H. (2018). Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications. Proceedings of the 2018 World Wide Web Conference.
></div>
>
><div id="ref10">
>[10] Zhang, C., Song, D., Chen, Y., Feng, X., Lumezanu, C., Cheng, W., Ni, J., Zong, B., Chen, H., & Chawla, N. V. (2019). A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 1409-1416.
></div>
>
><div id="ref11">
>[11] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. 2019. Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19). Association for Computing Machinery, New York, NY, USA, 2828–2837.
></div>
>
><div id="ref12">
>[12] Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, and Maria A. Zuluaga. 2020. USAD: UnSupervised Anomaly Detection on Multivariate Time Series. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 3395–3404.
></div>
><div id="ref13">
>[13] Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. Neurocomputing, Available online 2 June 2017, ISSN 0925-2312, https://doi.org/10.1016/j.neucom.2017.04.070
></div>
><div id="ref14">
>[14] Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. 2018. Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18). Association for Computing Machinery, New York, NY, USA, 387–395.
></div>
><div id="ref15">
>[15] Iurii D. Katser and Vyacheslav O. Kozitsin. Skoltech Anomaly Benchmark (SKAB). Kaggle, 2020. doi: 10.34740/KAGGLE/DSV/1693952.
></div>
><div id="ref16">
>[16] Adepu S., Junejo K. N., and Mathur A., “A Dataset to Support Research in the Design of Secure Water Treatment Systems,” The 11th International Conference on Critical Information Infrastructures Security.
></div>
><div id="ref17">
>[17] Iman Sharafaldin, Arash Habibi Lashkari, and Ali A Ghorbani. Toward generating a new intrusion detection dataset and intrusion traffic characterization. In ICISSp, pages 108–116, 2018.
></div>
><div id="ref18">
>[18] Andrea Dal Pozzolo, Olivier Caelen, Reid A Johnson, and Gianluca Bontempi. Calibrating probability with undersampling for unbalanced classification. In 2015 IEEE Symposium Series on Computational Intelligence, pages 159–166. IEEE, 2015.
></div>
><div id="ref19">
>[19] Steffen Moritz, Frederik Rehbach, Sowmya Chandrasekaran, Margarita Rebolledo, and Thomas Bartz-Beielstein. Gecco industrial challenge 2018 dataset: A water quality dataset for the ’internet of things: Online anomaly detection for drinking water quality’ competition at the genetic and evolutionary computation conference 2018, kyoto, japan., Feb 2018.
></div>
><div id="ref20">
>[20] Rafal Angryk, Petrus Martens, Berkay Aydin, Dustin Kempton, Sushant Mahajan, Sunitha Basodi, Azim Ahmadzadeh, Xumin Cai, Soukaina Filali Boubrahimi, Shah Muhammad Hamdi, Micheal Schuh, and Manolis Georgoulis. SWAN-SF, 2020.
></div>
