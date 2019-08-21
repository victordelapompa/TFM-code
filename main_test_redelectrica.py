# Authors: Victor de la Pompa <vpp020294@gmail.com>
#
#
# Red electrica: GPR
import numpy as np
import pandas as pd
import sys

from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, RationalQuadratic, Matern

import algoritmos
import lector


carpeta = '/gaa/home/catedra/wind/'

k1 = 1 * RBF(1) + WhiteKernel(0.1)
k2 = 1 * RBF(1) + 1 * RBF(100) + WhiteKernel(0.1)
k3 = 1 * RationalQuadratic(1, 1) + WhiteKernel(0.1)
k4 = 1 * Matern(1, nu=0.5) + WhiteKernel(0.1)
k5 = 1 * Matern(1, nu=1.5) + WhiteKernel(0.1)
k6 = 1 * Matern(1, nu=2.5) + WhiteKernel(0.1)

X_train, y_train, X_test, y_test, t = lector.leer_ree(val = False, reduced = int(sys.argv[2]) == 1, carpeta1 = carpeta, carpeta2 = '')
nombre = 'resultados/predicciones_redelectrica/gp_'
if int(sys.argv[1]) == 1:
    kernels = [k1, k2]
    nombres = [nombre + 'rbf', nombre +'2rbf']
    print("RBF y 2 RBF")
elif int(sys.argv[1]) == 2:
    kernels = [k3, k4]
    nombres = [nombre + 'rq', nombre +'m05']
    print("RQ y MATERN 0.5")
else:
    kernels = [k5, k6]
    print("MATERN 1.5 y 2.5")
    nombres = [nombre +'m15', nombre + 'm25']
algoritmos.gp_test(kernels, nombres, X_train, y_train, X_test, y_test, t)

