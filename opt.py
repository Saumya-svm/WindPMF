import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import time
import pickle
from random import shuffle
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, UpSampling2D
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
import numpy as np
from keras.callbacks import LearningRateScheduler
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_objective
from skopt.utils import use_named_args, dump, load
from keras.callbacks import TensorBoard
import argparse
import shutil
from train import prepare_model
from utils import prepare_data

def log_dir_name(learning_rate, decay_rate, step_size, seq_len, numBlocks, numLayers, filters):
    s = "./19_logs/lr_{0:.0e}_decay_rate_{1}_step_size_{2}_seq_len_{3}_numBlocks_{4}_num_layers_{5}_filters_{6}_/"
    log_dir = s.format(learning_rate, decay_rate, step_size, seq_len, numBlocks, numLayers, filters)

    return log_dir


learning_rate = Real(low=5e-4, high=1e-1, prior='log-uniform',name='learning_rate')
decay_rate = Real(low=0.1, high=0.98, name='decay_rate')
step_size = Integer(low=10, high=100, name='step_size')
seq_len = Categorical([4,8,16,32], name='seq_len')
numBlocks = Categorical([2,3,4,5], name='numBlocks')
numLayers = Categorical([2,3,4,5], name='numLayers')
filters = Categorical([16,32,64,128,256], name='filters')
dimensions = [learning_rate, decay_rate, step_size, seq_len, numBlocks, numLayers, filters]

path_best_model = 'best_model.h5'
best_mse = 0.0
j = 0

import os
def save_to_drive():
  if not os.path.exists('drive/MyDrive/tcnn_logs'):
    os.mkdir('drive/MyDrive/tcnn_logs')

  files = os.listdir('.')
  for f in files:
    if f.startswith("training_log_"):
      shutil.copy(f, 'drive/MyDrive/tcnn_logs')

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, decay_rate,
            step_size, seq_len, numBlocks, numLayers, filters):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    decay_rate:        Decay rate for the learning rate.
    step_size:         Step size for the learning rate decay.
    seq_len:           Length of the sequence.
    numBlocks:         Number of blocks in the model.
    numLayers:         Number of layers in each block.
    filters:           Number of filters in the convolutional layers.
    """

    print('learning rate: {0:.1e}'.format(learning_rate))
    print('decay rate:', decay_rate)
    print('step size:', step_size)
    print('sequence length:', seq_len)
    print('number of blocks:', numBlocks)
    print('number of layers:', numLayers)
    print('filters:', filters)
    print()

    global j
    j += 1
    print(j)

    trainX, trainY, valX, valY, testX, testY = prepare_data(int(seq_len))

    model, lr_sched = prepare_model(learning_rate,
                         decay_rate, step_size, seq_len, numBlocks, numLayers, filters)

    log_dir = log_dir_name(learning_rate, decay_rate,
            step_size, seq_len, numBlocks, numLayers, filters)

    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False)

    class LossHistory(tf.keras.callbacks.Callback):
      def __init__(self, run):
            super().__init__()
            self.run = run

      def on_epoch_end(self, epoch, logs=None):
          with open(f'training_log_{self.run}.txt', 'a') as f:
              f.write(f"Epoch {epoch + 1}, Loss: {logs['loss']}, MSE: {logs['mean_squared_error']}, "
                      f"Val_Loss: {logs['val_loss']}, Val_MSE: {logs['val_mean_squared_error']}\n")

    loss_history = LossHistory(j)

    seq_len = int(seq_len)
    num_samples = trainX.shape[0]
    val_num_samples = valX.shape[0]
    model(trainX)
    start = time.time()
    model_fit = model.fit(tf.convert_to_tensor(trainX, dtype=tf.float32), tf.convert_to_tensor(trainY, dtype=tf.float32 ), epochs=500, batch_size=64, steps_per_epoch=num_samples//64,validation_data=(tf.convert_to_tensor(valX, dtype=tf.float32), tf.convert_to_tensor(valY, dtype=tf.float32)), callbacks=[lr_sched, loss_history], validation_steps=val_num_samples//64)
    end = time.time()
    print("Time required to train one model(in s) : ", end-start)
    history = (model_fit)

    mse = history.history['val_mean_squared_error'][-1]

    print()
    print("mse: ", mse)
    print()

    save_to_drive()

    global best_mse

    if mse < best_mse:
        model.save(path_best_model)
        best_mse = mse
    K.clear_session()
    return mse