import numpy as np


from .__simple_features import get_log_feature, get_diff_feature, get_diff2_feature
from .__window_features import get_window_feature
from .__decomposition_features import get_stl_feature
from .__frequency_features import get_sr_feature, get_wavelet_feature
from .__regression_features import get_sarima_feature, get_addes_feature, get_simplees_feature, get_holt_feature


class FeatureExtractor(object):
    def __init__(self):
        self.__feature_buffer = None

    def __size_align(self):
        # TODO
        pass

    def extract_feature(self):
        pass

    def feature_summary(self):
        # TODO
        pass

    def feature_visualize(self):
        # TODO
        pass
