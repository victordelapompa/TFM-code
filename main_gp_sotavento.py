import numpy as np
import pandas as pd

import algoritmos
import lector

from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, RationalQuadratic, Matern


k1 = 1 * RBF(1) + WhiteKernel(0.1)
k2 = 1 * RBF(1) + 1 * RBF(27.4) + WhiteKernel(0.1)
k3 = 1 * RationalQuadratic(71, 0.0158) + WhiteKernel(0.1)
k4 = 1 * Matern(1, nu=0.5) + WhiteKernel(0.1)
k5 = 1 * Matern(1, nu=1.5) + WhiteKernel(0.1)
k6 = 1 * Matern(1, nu=2.5) + WhiteKernel(0.1)

X_train, y_train, X_test, y_test = lector.leer_sotavento(val = True, carpeta = '')

error_list = algoritmos.gp_val([k1, k2, k3, k4, k5, k6], X_train, y_train, X_test, y_test)
print("El que ha dado menor mae clipped ha sido el kernel:", np.argmin(error_list) + 1, " con error: ", np.min(error_list))

X_train, y_train, X_test, y_test = lector.leer_sotavento(val = False, carpeta = '')

nombre = 'resultados/predicciones_sotavento/gp_'
nombres = [nombre + 'rbf', nombre + '2rbf', nombre + 'rq', nombre + 'm05', nombre + 'm15', nombre + 'm25']

error_list = algoritmos.gp_test([k1, k2, k3, k4, k5, k6], nombres, X_train, y_train, X_test, y_test, range(len(y_test)))
print("El que ha dado menor mae clipped ha sido el kernel:", np.argmin(error_list) + 1, " con error: ", np.min(error_list))
