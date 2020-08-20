from statsmodels.tsa.api import SARIMAX, ExponentialSmoothing, SimpleExpSmoothing, Holt


def get_sarima_feature(value):
    predict = SARIMAX(value,
                      trend='n').fit(disp=0).get_prediction()
    return value - predict.predicted_mean


def get_addes_feature(value):
    predict = ExponentialSmoothing(value, trend='add').fit(smoothing_level=1)
    return value - predict.fittedvalues


def get_simplees_feature(value):
    predict = SimpleExpSmoothing(value).fit(smoothing_level=1)
    return value - predict.fittedvalues


def get_holt_feature(value):
    predict = Holt(value).fit(smoothing_level=1)
    return value - predict.fittedvalues
