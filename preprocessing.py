""" CIFAR Mean/Std Preprocessing
    Competitive Residual Network (CoRN) for image classification
    Last modified: 7/2/2018
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

class preprocess(object):

    def __int__(self):
        self.mean_r = 0
        self.mean_g = 0
        self.mean_b = 0
        self.std_r = 0
        self.std_g = 0
        self.std_b = 0

    def fit(self, x):

        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))

        scale = float(x.shape[0] * x.shape[1] * x.shape[2])
        self.mean_r = np.sum(x[:, :, :, 0]) / scale
        self.mean_g = np.sum(x[:, :, :, 1]) / scale
        self.mean_b = np.sum(x[:, :, :, 2]) / scale
        self.std_r = np.sqrt(np.sum(np.square(x[:, :, :, 0] - self.mean_r)) / (scale - 1))
        self.std_g = np.sqrt(np.sum(np.square(x[:, :, :, 1] - self.mean_g)) / (scale - 1))
        self.std_b = np.sqrt(np.sum(np.square(x[:, :, :, 2] - self.mean_b)) / (scale - 1))

    def apply(self, x):
        if x.ndim != 4:
            raise ValueError('Input to `.apply()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))

        x[:, :, :, 0] = (x[:, :, :, 0] - self.mean_r) / self.std_r
        x[:, :, :, 1] = (x[:, :, :, 1] - self.mean_g) / self.std_g
        x[:, :, :, 2] = (x[:, :, :, 2] - self.mean_b) / self.std_b

        return x

    def reverse(self, x):
        if x.ndim != 4:
            raise ValueError('Input to `.apply()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))

        x[:, :, :, 0] = x[:, :, :, 0]*self.std_r + self.mean_r
        x[:, :, :, 1] = x[:, :, :, 1]*self.std_g + self.mean_g
        x[:, :, :, 2] = x[:, :, :, 2]*self.std_b + self.mean_b

        return x
