import os
import pdb
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from model.LightGBM import LightGBM
from model.XGBoost import XGBoost
from model.RandomForest import RandomForest
from model.DNN import DNN

from metrics import modified_precision, modified_recall, modified_f1
from config import *


def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='KPI Anomaly Detection')

    parser.add_argument("--model", dest='model', type=str, required=True, 
        choices=['xgboost', 'lightgbm', 'random_forest', 'dnn'], help='The model used in the experiment')
    parser.add_argument("--train-path", dest='train_path', type=str, default='data/train/', help='The path of trianing dataset')
    parser.add_argument("--test-path", dest='test_path', type=str, default='data/test/', help='The path of testing dataset')
    parser.add_argument("--ngpu", dest='num_gpu', help="The number of gpu to use", default=1, type=int)
    parser.add_argument("--seed", dest='seed', type=int, default=2019, help="The random seed") 

    parser.add_argument("--threshold", dest='threshold', type=float, default=0.5, help='The threshold for anomaly score calculation')
    parser.add_argument("--delay", dest="delay", type=int, default=7, help='The delay of tolerance')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()

    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    gpu_memory = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu_ids = np.argsort(-1*np.array(gpu_memory))
    os.system('rm tmp')
    assert(args.num_gpu <= len(gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str, gpu_ids[:args.num_gpu]))
    print('Current GPU [%s], free memory: [%s] MB'%(os.environ['CUDA_VISIBLE_DEVICES'], ','.join(map(str, np.array(gpu_memory)[gpu_ids[:args.num_gpu]]))))

    # Set the random seed
    np.random.seed(args.seed)

    if args.model == 'xgboost':
        print('\033[0;34m%s\033[0m'%'Using model XGBoost...')
        model = XGBoost()
    elif args.model == 'lightgbm':
        print('\033[0;34m%s\033[0m'%'Using model LightGBM...')
        model = LightGBM()
    elif args.model == 'random_forest':
        print('\033[0;34m%s\033[0m'%'Using model RandomForest...')
        model = RandomForest()
    elif args.model == 'dnn':
        print('\033[0;34m%s\033[0m'%'Using model DNN...')
        model = DNN(input_size=DNN_NUM_FEATURE)
    else:
        raise NotImplementedError('\033[0;31m%s\033[0m'%'Invalid model name!')

    train_files = os.listdir(args.train_path)

    # train_data = pd.read_csv(args.train_path+train_files[0])
    # train_data.fillna(0, inplace=True)
    # train_data = train_data.values
    train_data = []
    for i in tqdm(range(len(train_files)), desc='READ_TRAIN'):
        temp_data = pd.read_csv(args.train_path+train_files[i])
        temp_data.fillna(0, inplace=True)
        temp_data = temp_data.values
        train_data.append(temp_data)
    train_data = np.concatenate(train_data, axis=0)
    x_train, y_train = train_data[:,:-1], train_data[:,-1]

    print('\033[0;34m%s\033[0m'%'Start training...')
    model.fit(x_train, y_train)

    print('\033[0;34m%s\033[0m'%'Start evaluating...')
    test_files = os.listdir(args.test_path)
    y_preds = []
    y_trues = []
    for i in tqdm(range(len(test_files)), desc='READ_TEST'):
        test_data = pd.read_csv(args.test_path+test_files[i])
        test_data.fillna(0, inplace=True)
        x_test, y_test = test_data.iloc[:,:-1].values, test_data.iloc[:,-1].values

        y_pred = model.predict(x_test)
        y_pred[y_pred>args.threshold] = 1
        y_pred[y_pred<=args.threshold] = 0
        y_preds.append(y_pred)
        y_trues.append(y_test)

    print('Precision: ', modified_precision(y_preds, y_trues, args.delay))
    print('Recall: ', modified_recall(y_preds, y_trues, args.delay))
    print('F1: ', modified_f1(y_preds, y_trues, args.delay))
