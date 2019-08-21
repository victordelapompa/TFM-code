# Authors: Victor de la Pompa <vpp020294@gmail.com>
#
#
# SVR Matern

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import Matern
from copy import deepcopy

class MaternSVR(SVR, RegressorMixin):
    def __init__(self, length_scale = 1.0, nu = 0.5, C=1.0, epsilon=0.1):
        self.length_scale = length_scale
        self.gamma = (1.0/length_scale)**2
        self.C = C
        self.nu = nu
        self.epsilon = epsilon

    def fit(self, X, y, sample_weight=None):
        # Para que funcione el gridSearchCV el kernel se tiene que iniciar aqui
        self.X = X
        self.svr = SVR('precomputed', C = self.C, epsilon = self.epsilon)
        self.kernel = Matern(self.length_scale, nu = self.nu)
        K = self.kernel.__call__(X = X, Y = None, eval_gradient = False)
        self.svr.fit(K, y, sample_weight)
        return self

    def predict(self,  X):
        K = self.kernel.__call__(X = X, Y = self.X, eval_gradient = False)
        return self.svr.predict(K)