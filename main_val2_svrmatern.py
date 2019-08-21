import numpy as np

import algoritmos
import lector
import sys

X_train, y_train, X_test, y_test = lector.leer_sotavento(val = True, carpeta = '')

algoritmos.svm_matern_val_C_epsilon(X_train, y_train, X_test, y_test, 331,'sotavento')
print("Sotavento finito")