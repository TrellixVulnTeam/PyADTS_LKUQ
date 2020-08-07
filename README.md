# KPI Anomaly Detection

## Introduction

This project contains multiple KPI anomaly detection algorithms.

## Usage

To run an algorithm, launch `main.py` using your Python interpreter with arguments as follows:

```bash
usage: main.py [-h] --model {xgboost,lightgbm,random_forest,dnn}
               [--train-path TRAIN_PATH] [--test-path TEST_PATH]
               [--ngpu NUM_GPU] [--seed SEED] [--threshold THRESHOLD]
               [--delay DELAY]

KPI Anomaly Detection

optional arguments:
  -h, --help            show this help message and exit
  --model {xgboost,lightgbm,random_forest,dnn}
                        The model used in the experiment
  --train-path TRAIN_PATH
                        The path of trianing dataset
  --test-path TEST_PATH
                        The path of testing dataset
  --ngpu NUM_GPU        The number of gpu to use
  --seed SEED           The random seed
  --threshold THRESHOLD
                        The threshold for anomaly score calculation
  --delay DELAY         The delay of tolerance
```
