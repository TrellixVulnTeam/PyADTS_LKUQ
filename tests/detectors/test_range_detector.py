"""
@Time    : 2021/10/25 11:22
@File    : test_range_detector.py
@Software: PyCharm
@Desc    : 
"""
import os

import numpy as np

from pyadts.detectors import RangeDetector

detector = RangeDetector(low=0, high=2)
x = np.random.randn(100)
y = np.random.randint(low=0, high=2, size=(100,))


def test_fit():
    detector.fit(x, y)
    print(detector.__dict__)
    detector.save('./test.pkl')


def test_predict():
    detector.predict(x)
    detector.load('./test.pkl')


def test_score():
    detector.score(x)
    os.remove('./test.pkl')
