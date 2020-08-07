import os
import pdb

import pywt
import numpy as np
import pandas as pd

from tqdm import tqdm

from statsmodels.tsa.api import SARIMAX, ExponentialSmoothing, SimpleExpSmoothing, Holt

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureExtractor(object):
    def __init__(self, window_size_list, eps=1e-8, mag_window=3, verbose=True, normalize='z_score'):
        self.window_size_list = window_size_list
        self.eps = eps
        self.mag_window = mag_window
        self.verbose = verbose
        self.normalize = normalize

        if normalize == 'none':
            self.scaler = None
        elif normalize == 'min_max':
            self.scaler = MinMaxScaler()
        elif normalize == 'z_score':
            self.scaler = StandardScaler()
        else:
            raise NotImplementedError('Invalid normalization method!')

    def extract(self, time_series):
        if self.scaler is not None:
            time_series = self.scaler.fit_transform(time_series.reshape(-1,1)).reshape(-1)

        # feature_log = self.__get_feature_logs(time_series) # Delete log features
        feature_wavelet = self.__get_feature_wavelet(time_series, ['db2'])
        feature_SARIMA_residuals = self.__get_feature_SARIMA_residuals(time_series)
        feature_AddES_residuals = self.__get_feature_AddES_residuals(time_series)
        feature_SimpleES_residuals = self.__get_feature_SimpleES_residuals(time_series)
        feature_Holt_residuals = self.__get_feature_Holt_residuals(time_series)
        feature_window = self.__get_feature_window(time_series, self.window_size_list)
        feature_spectral_residual = self.__spectral_residual_transform(time_series)

        feature_window_length = feature_window.shape[0]
        # feature_log = feature_log[-feature_window_length:].reshape(-1, 1)
        feature_wavelet = feature_wavelet[-feature_window_length:,:]
        feature_SARIMA_residuals = feature_SARIMA_residuals[-feature_window_length:].reshape(-1, 1)
        feature_AddES_residuals = feature_AddES_residuals[-feature_window_length:].reshape(-1, 1)
        feature_SimpleES_residuals = feature_SimpleES_residuals[-feature_window_length:].reshape(-1, 1)
        feature_Holt_residuals = feature_Holt_residuals[-feature_window_length:].reshape(-1, 1)
        feature_spectral_residual = feature_spectral_residual[-feature_window_length:].reshape(-1, 1)

        features = np.concatenate((feature_wavelet, feature_SARIMA_residuals, feature_AddES_residuals, feature_SimpleES_residuals, feature_Holt_residuals, feature_window, feature_spectral_residual), axis=1)

        return features

    def __get_feature_logs(self, time_series):
        return np.log(time_series + 1e-2)

    def __get_feature_SARIMA_residuals(self, time_series):
        predict = SARIMAX(time_series,
                          trend='n').fit(disp=0).get_prediction()
        return time_series - predict.predicted_mean

    def __get_feature_AddES_residuals(self, time_series):
        predict = ExponentialSmoothing(time_series, trend='add').fit(smoothing_level=1)
        return time_series - predict.fittedvalues

    def __get_feature_SimpleES_residuals(self, time_series):
        predict = SimpleExpSmoothing(time_series).fit(smoothing_level=1)
        return time_series - predict.fittedvalues

    def __get_feature_Holt_residuals(self, time_series):
        predict = Holt(time_series).fit(smoothing_level=1)
        return time_series - predict.fittedvalues

    def __get_feature_window(self, time_series, window_size_list):
        start_point = 2*max(window_size_list)
        start_accum = 0
        data = []

        for i in tqdm(np.arange(start_point, len(time_series)), desc='SLIDING_WINDOW'):        
            # the datum to put into the data pool
            datum = []

            # fill the datum with f01-f09
            diff_plain = time_series[i] - time_series[i-1]
            start_accum = start_accum + time_series[i]
            mean_accum = (start_accum)/(i-start_point+1)

            # f06: diff
            datum.append(diff_plain)
            # f07: diff percentage
            datum.append(diff_plain/(time_series[i-1] + 1e-10))  # to avoid 0, plus 1e-10
            # f08: diff of diff - derivative
            datum.append(diff_plain - (time_series[i-1] - time_series[i-2]))
            # f09: diff of accumulated mean and current value
            datum.append(time_series[i] - mean_accum)

            # fill the datum with features related to windows
            # loop over different windows size to fill the datum
            for k in window_size_list:
                mean_w = np.mean(time_series[i-k:i+1])
                var_w = np.mean((np.asarray(time_series[i-k:i+1]) - mean_w)**2)
                #var_w = np.var(time_series[i-k:i+1])

                mean_w_and_1 = mean_w + (time_series[i-k-1]-time_series[i])/(k+1)
                var_w_and_1 = np.mean((np.asarray(time_series[i-k-1:i]) - mean_w_and_1)**2)
                #mean_w_and_1 = np.mean(time_series[i-k-1:i])
                #var_w_and_1 = np.var(time_series[i-k-1:i])

                mean_2w = np.mean(time_series[i-2*k:i-k+1])
                var_2w = np.mean((np.asarray(time_series[i-2*k:i-k+1]) - mean_2w)**2)
                #var_2w = np.var(time_series[i-2*k:i-k+1])

                # diff of sliding windows
                diff_mean_1 = mean_w - mean_w_and_1
                diff_var_1 = var_w - var_w_and_1

                # diff of jumping windows
                diff_mean_w = mean_w - mean_2w
                diff_var_w = var_w - var_2w

                # f1
                datum.append(mean_w)  # [0:2] is [0,1]
                # f2
                datum.append(var_w)
                # f3
                datum.append(diff_mean_1)
                # f4
                datum.append(diff_mean_1/(mean_w_and_1 + 1e-10))
                # f5
                datum.append(diff_var_1)
                # f6
                datum.append(diff_var_1/(var_w_and_1 + 1e-10))
                # f7
                datum.append(diff_mean_w)
                # f8
                datum.append(diff_mean_w/(mean_2w + 1e-10))
                # f9
                datum.append(diff_var_w)
                # f10
                datum.append(diff_var_w/(var_2w + 1e-10))

                # diff of sliding/jumping windows and current value
                # f11
                datum.append(time_series[i] - mean_w_and_1)
                # f12
                datum.append(time_series[i] - mean_2w)

            data.append(np.asarray(datum))

        return np.asarray(data)

    def __average_filter(self, values, n=3):
        """
        Calculate the sliding window average for the give time series.
        Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
        :param values: list.
            a list of float numbers
        :param n: int, default 3.
            window size.
        :return res: list.
            a list of value after the average_filter process.
        """

        if n >= len(values):
            n = len(values)

        res = np.cumsum(values, dtype=float)
        res[n:] = res[n:] - res[:-n]
        res[n:] = res[n:] / n

        for i in range(1, n):
            res[i] /= (i + 1)

        return res

    def __time_series_fft(self, values):
        trans = np.fft.fft(values)
        freq = np.fft.fftfreq(values.shape[-1])
        return trans, freq

    def __spectral_residual_transform(self, values):
        """
        This method transform a time series into spectral residual series
        :param values: list.
            a list of float values.
        :return: mag: list.
            a list of float values as the spectral residual values
        """

        trans = np.fft.fft(values)
        mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
        eps_index = np.where(mag <= self.eps)[0]
        mag[eps_index] = self.eps

        mag_log = np.log(mag)
        mag_log[eps_index] = 0

        spectral = np.exp(mag_log - self.__average_filter(mag_log, n=self.mag_window))

        trans.real = trans.real * spectral / mag
        trans.imag = trans.imag * spectral / mag
        trans.real[eps_index] = 0
        trans.imag[eps_index] = 0

        wave_r = np.fft.ifft(trans)
        mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
        return mag

    def __wavelet_decomposition(self, data, w, level=5):
        w = pywt.Wavelet(w)
        mode = pywt.Modes.smooth
        a = data
        ca = []
        cd = [] 
        for i in range(level):
            (a, d) = pywt.dwt(a, w, mode)
            ca.append(a)
            cd.append(d)

        rec_a = []
        rec_d = []

        for i in range(level):
            rec_a.append(pywt.upcoef('a', ca[i], w, level=i+1, take=data.shape[0]))
            rec_d.append(pywt.upcoef('d', cd[i], w, level=i+1, take=data.shape[0]))

        return np.transpose(np.concatenate((np.array(rec_a), np.array(rec_d)), axis=0))

    def __get_feature_wavelet(self, time_series, wavelet_names=['db2']):
        wavelet_features = []
        for w in wavelet_names:
            wavelet_features.append(self.__wavelet_decomposition(time_series, w))
        
        return np.concatenate(wavelet_features, axis=1)

if __name__ == '__main__':
    NORMALIZE = 'z_score'

    print('\033[0;34m%s\033[0m'%'Extracting features for the training dataset...')
    with open('../data/phase2_train.csv', 'r', encoding='utf8') as f:
        df_train = pd.read_csv(f)

    if not os.path.exists('../data/train'):
        os.makedirs('../data/train')

    feature_extractor = FeatureExtractor([2, 5, 10, 25, 50, 100], normalize=NORMALIZE)
    for name, data in tqdm(df_train.groupby('KPI ID'), desc='KPI_ID'):
        features = feature_extractor.extract(data['value'].values)
        features = pd.DataFrame(np.concatenate((data['value'].values[-features.shape[0]:].reshape(-1, 1), features, data['label'].values[-features.shape[0]:].reshape(-1, 1)), axis=1), columns=['feature_raw']+['feature%d'%i for i in range(features.shape[1])]+['label'])

        features.to_csv('../data/train/ID_'+name+'.csv', index=False)

    print('\033[0;34m%s\033[0m'%'Extracting features for the testing dataset...')
    df_test = pd.read_hdf('../data/phase2_ground_truth.hdf')

    if not os.path.exists('../data/test'):
        os.makedirs('../data/test')

    feature_extractor = FeatureExtractor([2, 5, 10, 25, 50, 100], normalize=NORMALIZE)
    for name, data in tqdm(df_test.groupby('KPI ID'), desc='KPI_ID'):
        features = feature_extractor.extract(data['value'].values)
        features = pd.DataFrame(np.concatenate((data['value'].values[-features.shape[0]:].reshape(-1, 1), features, data['label'].values[-features.shape[0]:].reshape(-1, 1)), axis=1), columns=['feature_raw']+['feature%d'%i for i in range(features.shape[1])]+['label'])

        features.to_csv('../data/test/ID_'+str(name)+'.csv', index=False)
