import numpy as np

import algoritmos
import lector
import sys

X_train, y_train, X_test, y_test = lector.leer_sotavento(val = True, carpeta = '')

algoritmos.svm_matern_val(X_train, y_train, X_test, y_test, 'sotavento')
print("Sotavento finito")
sys.stdout.flush()
carpeta = '/gaa/home/catedra/wind/'
X_train, y_train, X_test, y_test, t = lector.leer_ree(val = True, reduced = True, carpeta1 = carpeta, carpeta2 = '')
algoritmos.svm_matern_val(X_train, y_train, X_test, y_test, 'ree')
