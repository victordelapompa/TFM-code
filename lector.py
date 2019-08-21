# Authors: Victor de la Pompa <vpp020294@gmail.com>
#
#
# Leer ficheros red_electrica o sotavento
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Devuelve los conjuntos de entrenamiento y test para red electrica.
# En caso de val = True, el conjunto de test es el de validación (2 año)
def leer_ree(val = True, reduced = False, carpeta1 = '/gaa/home/catedra/wind/', carpeta2 = ''):
    '''

    LECTURA + JUNTAR LA PRODUCCION CON LA MATRIZ DE DATOS

    '''
    df = pd.read_csv(carpeta2 + 'prods_ree_20130101_20170406_corrected.csv', sep=',')
    df.columns = ['time', 'prod']
    if reduced: # QUEDARME SOLO CON 0.5
        f = np.load(carpeta2 + "ree_reducida.npy")
    else:
        f = np.load(carpeta1 + '2013010100_2015123100.mdata.npy')

    pat = '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})'
    df['time'] = pd.to_datetime(df['time'].astype('str').str.extract(pat, expand=True))
    df['time'] = [1000000*dt_time.year + 10000*dt_time.month + 100*dt_time.day + dt_time.hour for dt_time in df.time]
    df.drop_duplicates(subset = ['time'], keep='first', inplace = True)
    df.reset_index(drop=True, inplace=True)

    aux = np.in1d(df.time, f[:,0])
    t = np.intersect1d(df.time, f[:,0])
    y = df['prod'][aux]

    t_ind = np.in1d(f[:,0], df.time)
    X = f[t_ind,1:]


    '''

    NORMALIZACION + SEPARACION TRAIN Y TEST

    '''

    if val:
        val_maxindex = np.argmax(t[t <= 2014123123]) + 1 
        X = X[:val_maxindex, :]
        y = y[:val_maxindex]
        t = t[:val_maxindex]
        train_maxindex = np.argmax(t[t <= 2013123123]) + 1 
    else:
        train_maxindex = np.argmax(t[t <= 2014123123]) + 1 

    X_train = X[:train_maxindex, :]
    y_train = y[:train_maxindex]

    X_test = X[train_maxindex:, :]
    y_test = y[train_maxindex:]
    scaler = preprocessing.StandardScaler(copy = False).fit(X_train)

    return scaler.transform(X_train), y_train, scaler.transform(X_test), y_test, t[train_maxindex:]

# Devuelve los conjuntos de entrenamiento y test para sotavento.
# En caso de val = True, el conjunto de test es el de validación (2 año)
def leer_sotavento(val = True, carpeta = ''):
    sotavento_train = np.load(carpeta + "stv_3h.train.npy")
    X_train = sotavento_train[:, 1:]
    y_train = sotavento_train[:, 0]

    sotavento_val = np.load(carpeta + "stv_3h.val.npy")
    X_val = sotavento_val[:, 1:]
    y_val = sotavento_val[:, 0]
    if val:
        return X_train, y_train, X_val,  y_val

    X_train = np.concatenate((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))

    sotavento_test = np.load(carpeta + "stv_3h.test.npy")
    X_test = sotavento_test[:, 1:]
    y_test = sotavento_test[:, 0]

    return X_train, y_train, X_test, y_test

# Dato cada hora en sotavento
def leer_sotavento2(val = True, carpeta = '/gaa/home/catedra/wind/'):
    sotavento_train = np.load(carpeta + "stv_h_ext.train.npy")
    X_train = sotavento_train[:, 1:]
    y_train = sotavento_train[:, 0]

    sotavento_val = np.load(carpeta + "stv_h_ext.val.npy")
    X_val = sotavento_val[:, 1:]
    y_val = sotavento_val[:, 0]
    if val:
        return X_train, y_train, X_val,  y_val

    X_train = np.concatenate((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))

    sotavento_test = np.load(carpeta + "stv_h_ext.test.npy")
    X_test = sotavento_test[:, 1:]
    y_test = sotavento_test[:, 0]

    return X_train, y_train, X_test, y_test


# Ficheros ya reducidos
def leer_ree2(val = True, carpeta = '/gaa/home/catedra/wind/'):
    ree_train = np.load(carpeta + "pen_super_reduced_h.train.npy")
    X_train = ree_train[:, 1:]
    y_train = ree_train[:, 0]

    ree_val = np.load(carpeta + "pen_super_reduced_h.val.npy")
    X_val = ree_val[:, 1:]
    y_val = ree_val[:, 0]
    if val:
        return X_train, y_train, X_val, y_val

    X_train = np.concatenate((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))

    ree_test = np.load(carpeta + "pen_super_reduced_h.test.npy")
    X_test = ree_test[:, 1:]
    y_test = ree_test[:, 0]



    return X_train, y_train, X_test, y_test
