import time

import numpy as np
import pickle 

import algoritmos
import lector

from svr_matern import MaternSVR

carpeta = '/gaa/home/catedra/wind/'
X_train, y_train, X_test, y_test, t = lector.leer_ree(val = False, reduced = True, carpeta1 = carpeta, carpeta2 = '')

f = open('resultados/grid_search_results_ree.pkl', mode='rb')
search = pickle.load(f)
f.close()

print("Experimento 1: ", search.best_params_)

y_pred = search.best_estimator_.predict(X_test)

mae, clipped_mae = algoritmos.calculate_error(y_test, y_pred)

print("|", mae, "|", clipped_mae)

f = open('resultados/grid_search_results_ree_lengthfixed.pkl', mode='rb')
search_lengthfixed = pickle.load(f)
f.close()

print("Experimento 4: ", search_lengthfixed.best_params_)

y_pred = search_lengthfixed.best_estimator_.predict(X_test)

mae, clipped_mae = algoritmos.calculate_error(y_test, y_pred)

print("|", mae, "|", clipped_mae)


