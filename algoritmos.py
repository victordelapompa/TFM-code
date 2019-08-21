# Authors: Victor de la Pompa <vpp020294@gmail.com>
#
#
# Algoritmos ML
import time
import numpy as np
import pandas as pd
import pickle
import sys
import gc

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from svr_matern import MaternSVR

# Calcula el MAE y el MAE clip
def calculate_error(y_test, y_pred):
    clipped_pred = np.minimum(np.maximum(y_pred, 0), 1)
    clipped_mae = mean_absolute_error(y_test, clipped_pred)
    return mean_absolute_error(y_test, y_pred), clipped_mae

# Calcula y pinta el tiempo, MAE, MAE clip
# al entrenar un GP con distinto tipo de kernels.
def gp_val(kernels, X_train, y_train, X_test, y_test, num_restarts = 5):
    error_list = []
    for kernel in kernels:
        gc.collect()
        sys.stdout.flush()
        # Instanciate a Gaussian Process model
        gp = GaussianProcessRegressor(kernel=kernel, alpha= 0, copy_X_train  = False,
                                          n_restarts_optimizer= num_restarts, optimizer = 'fmin_l_bfgs_b', normalize_y = True)
        inicio = time.time()
        gp.fit(X_train, y_train)
        y_pred = gp.predict(X_test, return_std = False)
        tiempo = time.time() - inicio
        mae, clipped_mae = calculate_error(y_test, y_pred)

        print("|" , gp.kernel_, "|", gp.kernel_.theta, "|")
        print("| %0.5f | %0.5f | %0.5f |" % (tiempo/3600, mae, clipped_mae))
        print("")
        error_list.append(clipped_mae)

    return error_list


# Calcula y pinta el tiempo, MAE, MAE clip,
# al entrenar un GP con distinto tipo de kernels.

# Además por cada kernel guarda en nombre[i].csv 
# el valor real, la prediccion, sigma y los índices (de tiempo)
def gp_test(kernels, nombres, X_train, y_train, X_test, y_test, t, num_restarts = 5):
    pat = '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})'
    df_res = pd.DataFrame(data = np.zeros((len(y_test), 4)))
    df_res.columns = ['time','value', 'prediction', 'deviation']
    df_res['time'] = t
    df_res['time'] = pd.to_datetime(df_res['time'].astype('str').str.extract(pat, expand=True))
    df_res['value'] = y_test
    for kernel, nombre in zip(kernels, nombres):
        gc.collect()
        sys.stdout.flush()
        # Instanciate a Gaussian Process model
        gp = GaussianProcessRegressor(kernel=kernel, alpha= 0, copy_X_train  = False,
                                          n_restarts_optimizer= num_restarts, optimizer = 'fmin_l_bfgs_b', normalize_y = True)
        inicio = time.time()
        gp.fit(X_train, y_train)
        y_pred, sigma = gp.predict(X_test, return_std = True)
        tiempo = time.time() - inicio
        mae, clipped_mae = calculate_error(y_test, y_pred)

        print("|" , gp.kernel_, "|", gp.kernel_.theta, "|")
        print("| %0.5f | %0.5f | %0.5f | %.3f |" % (tiempo/3600, mae, clipped_mae, gp.log_marginal_likelihood_value_))
        print("")
        df_res['prediction'] = y_pred
        df_res['deviation'] = sigma
        df_res.to_csv(nombre + ".csv", encoding='utf-8', index=False)

# Calcula y pinta el tiempo, MAE de los residuos, verosimilitud (para elegir que kernel)
# al entrenar un GP con distinto tipo de kernels. 
# y_train : Residuo en validacion del modelo (SVR, errores del segundo año entrenando con el primero)
# Se tienen dos tipos de residuos en test:
# y_test : Residuo del modelo (SVR) entrenando con el primer año.
# y_test2: Residuo del modelo (SVR) entrenando con los dos primeros años.
# Además por cada kernel guarda en nombre[i].csv 
# el valor real del residuo, la prediccion del residuo y sigma
def gp_test_errors(kernels, nombres, X_train, y_train, X_test, y_test, y_test2, num_restarts = 5):
    df_res = pd.DataFrame(data = np.zeros((len(y_test), 4)))
    df_res.columns = ['e_value1','e_value2', 'prediction', 'deviation']
    df_res['e_value1'] = y_test[:]
    df_res['e_value2'] = y_test2[:]
    df_res.to_csv(nombres[0] + ".csv", encoding='utf-8', index=False)
    
    for kernel, nombre in zip(kernels, nombres):
        # Instanciate a Gaussian Process model
        gp = GaussianProcessRegressor(kernel=kernel, alpha= 0, copy_X_train  = False,
                                          n_restarts_optimizer= num_restarts, optimizer = 'fmin_l_bfgs_b', normalize_y = True)
        gc.collect()
        sys.stdout.flush()
        inicio = time.time()
        gp.fit(X_train, y_train)
        y_pred, sigma = gp.predict(X_test, return_std = True)
        tiempo = time.time() - inicio
        mae = mean_absolute_error(y_test, y_pred)
        mae2 = mean_absolute_error(y_test2, y_pred)
        print("|" , gp.kernel_, "|", gp.kernel_.theta, "|")
        print("| %0.5f | %0.5f | %0.5f |  %.3f |" % (tiempo/3600, mae, mae2, gp.log_marginal_likelihood_value_))
        print("")
        df_res['prediction'] = y_pred
        df_res['deviation'] = sigma
        df_res.to_csv(nombre + ".csv", encoding='utf-8', index=False)
        



# Realiza la validación del length_scale (gamma = 1/length_scale**2), C y epsilon para
# un SVR con kernel Matern nu = 0.5.
def svm_matern_val(X_train, y_train, X_test, y_test, nombre, jobs = 4):
    N = len(X_train)
    M = len(X_test)
    idx = -1*np.ones(N+M)
    idx[N:] = 0
    ps = PredefinedSplit(test_fold=idx)

    l_epsilon = [4.0**(-k)*np.std(y_train) for k in np.arange(1,6)]
    l_C = [4.0**k for k in np.arange(-5,6)]
    l_gamma = np.array([1.0/X_train.shape[1]*4.0**k for k in np.arange(-3,2)])

    param_grid = {'length_scale': np.sqrt(1/l_gamma), 'epsilon': l_epsilon, 'C': l_C}

    grid = GridSearchCV(MaternSVR(nu = 0.5), param_grid=param_grid, cv=ps, scoring='neg_mean_absolute_error',
                                      n_jobs= jobs, verbose=0)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    grid = grid.fit(X, y)
    print(grid.best_params_)
    f = open('resultados/grid_search_results_' + nombre + '.pkl', mode='wb')
    pickle.dump(grid, f)
    f.close()
    sys.stdout.flush()


# Realiza la validación del length scale fijo, C y epsilon para
# un SVR con kernel Matern nu = 0.5.
def svm_matern_val_C_epsilon(X_train, y_train, X_test, y_test, length_scale, nombre, jobs = 4):
    N = len(X_train)
    M = len(X_test)
    idx = -1*np.ones(N+M)
    idx[N:] = 0
    ps = PredefinedSplit(test_fold=idx)

    l_epsilon = [4.0**(-k)*np.std(y_train) for k in np.arange(1,6)]
    l_C = [4.0**k for k in np.arange(-5,6)]

    param_grid = {'epsilon': l_epsilon, 'C': l_C}

    grid = GridSearchCV(MaternSVR(length_scale, nu = 0.5), param_grid=param_grid, cv=ps, scoring='neg_mean_absolute_error',
                                      n_jobs= jobs, verbose=0)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    grid = grid.fit(X, y)
    print(grid.best_params_)
    f = open('resultados/grid_search_results_' + nombre + '_lengthfixed.pkl', mode='wb')
    pickle.dump(grid, f)
    f.close()
    sys.stdout.flush()

