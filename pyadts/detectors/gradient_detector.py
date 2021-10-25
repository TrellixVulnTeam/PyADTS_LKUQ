"""
@Time    : 2021/10/25 11:03
@File    : gradient_detector.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from pyadts.generic import Model


class GradientDetector(Model):
    def __init__(self, max_gradient: float = np.inf):
        super(GradientDetector, self).__init__()

        self.max_gradient
