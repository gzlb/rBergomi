import numpy as np
import pandas as pd
import time
import logging
import tensorflow.keras as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from cnn.cnn import AbstractModelTrainer


from scipy.special import hyp2f1
from scipy import special
from fbm import FBM
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Report:
    df: pd.DataFrame
    loss_history: dict

def generating_H():
    """ Different samplers of H
    """

    return [0.1, 0.2, 0.3, 0.4, 0.5]


def joint_shuffle(a, b):
    """ Randomly shuffle two arrays jointly
    """
    assert len(a) == len(b)
    shuffle_idx = np.random.permutation(np.arange(len(a)))
    return a[shuffle_idx], b[shuffle_idx]


def lagged_mean(path, q, lag):
    """Computes the mean of the q-powered lagged path
    
    Parameters:
    -----------
        path: 1D-array, the path
        q: the power
        lag: number of steps to lag
    """
    return np.mean(np.power(np.abs(path[lag:]-path[:-lag]), q))

def least_squares_helper(q, logs, path):
    avgs = [lagged_mean(path, q, lag) for lag in range(1, 31)]
    avgs_log = np.log(avgs)
    return np.polyfit(logs, avgs_log, 1)[0]

def least_squares_path(path):
    powers = [0.5, 1.0, 1.5, 2.0, 3.0]
    logs = np.log(np.arange(1, 31))
    K_q = [least_squares_helper(q, logs, path) for q in powers]
    return np.polyfit(powers, K_q, 1)[:2]

def least_squares(paths):
    return np.apply_along_axis(least_squares_path, 1, paths.reshape(paths.shape[:2]))

def shuffle_split_data(paths, labels):
    """ Shuffles a dataset and splits it into train/val/test
    """
    paths, labels = joint_shuffle(paths, labels)
    paths = paths.reshape((paths.shape[0], paths.shape[1], 1))
    labels = labels.reshape((labels.shape[0], -1))
    
    X_train, X_test, Y_train, Y_test = train_test_split(paths, labels, test_size=0.3)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def prepare_data(Hs, method='rbergomi', n_paths=5000, n_timesteps=500, T=1, eta=1):
    """ Prepares dataset for neural network. 
    1. Generate paths
    2. Shuffle and split dataset into train/val/test
    """
    paths = np.zeros((n_paths*len(Hs), n_timesteps), dtype=float)
    labels = np.zeros(n_paths*len(Hs), dtype=float)
    

    for H_idx, H in enumerate(Hs):
        paths[H_idx*n_paths:(H_idx+1)*n_paths, :] = generating_rBergomi_path(n_paths, n_timesteps, H, T, eta)
        labels[H_idx*n_paths:(H_idx+1)*n_paths] = H

    
    return shuffle_split_data(paths, labels)




def benchmark(X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, trainer: AbstractModelTrainer, batch_size: int = 64, n_epochs: int = 30, verbose: int = 0, time_steps: list = [100, 200, 300, 400, 500]) -> Report:
    report = pd.DataFrame(columns=['RMSE(CNN)', 'MRE(CNN)', 'Training Time (seconds)', 'Test Time (seconds)', 'RMSE(LS)', 'MRE(LS)', 'Time (seconds)'], index=time_steps)
    report.index.name = 'Input length'
    loss_history = {}

    for n_timesteps in time_steps:
        logging.info(f"Performing paths of {n_timesteps} time steps")

        start_time = time.time()
        model = trainer.build_model()

        X_train_t = X_train[:, :n_timesteps]
        X_val_t = X_val[:, :n_timesteps]
        X_test_t = X_test[:, :n_timesteps]

        logging.info("Running CNN")
        early_stopping = EarlyStopping(monitor='val_loss', patience=n_epochs, verbose=verbose, mode='auto', restore_best_weights=False)
        cnn_train = model.fit(X_train_t, Y_train, batch_size=batch_size, epochs=n_epochs, verbose=verbose, validation_data=(X_val_t, Y_val), callbacks=[early_stopping])

        cnn_train_time = time.time() - start_time
        report.loc[n_timesteps, 'Training Time (seconds)'] = cnn_train_time
        start_time = time.time()

        cnn_pred = model.predict(X_test_t)
        cnn_test_time = time.time() - start_time
        report.loc[n_timesteps, 'Test Time (seconds)'] = cnn_test_time
        report.loc[n_timesteps, 'RMSE(CNN)'] = np.sqrt(np.mean(np.power(cnn_pred-Y_test, 2)))
        report.loc[n_timesteps, 'MRE(CNN)'] = np.mean(np.abs(cnn_pred-Y_test) / Y_test)

        loss_history[n_timesteps] = {
            'train_loss': cnn_train.history['loss'],
            'val_loss': cnn_train.history['val_loss']
        }

        logging.info("Running Least Squares")
        start_time = time.time()
        ls_pred = least_squares(X_test_t)[:, 0]
        report.loc[n_timesteps, 'Time (seconds)'] = time.time() - start_time
        report.loc[n_timesteps, 'RMSE(LS)'] = np.sqrt(np.mean(np.power(ls_pred-Y_test, 2)))
        report.loc[n_timesteps, 'MRE(LS)'] = np.mean(np.abs(ls_pred-Y_test) / Y_test)

    return Report(df=report, loss_history=loss_history)