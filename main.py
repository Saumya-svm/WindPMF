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
from utils import parse, prepare_data, prepare_model
from train import train_save_model
import pickle
from opt import fitness

def eval():
    with open('opt_params.pkl', 'rb') as f:
        optimized_params = pickle.load(f)

    trainX, trainY, valX, valY, testX, testY = prepare_data(int(optimized_params[3]))
    model, lr_sched = prepare_model(*optimized_params)
    model(trainX)
    model.load_weights('best_model')
    
    model.evaluate(trainX, trainY)
    model.evaluate(testX, testY)
    model.evaluate(valX, valY)


def train():
    learning_rate = args.ini_learning_rate
    decay_rate = args.decay_rate
    step_size = args.step_size
    seq_len = args.seq_len
    numBlocks = args.numBlocks
    numLayers = args.numLayers
    filters = args.filters

    optimized_params = [learning_rate, decay_rate, step_size, seq_len, numBlocks, numLayers, filters]

    train_save_model(optimized_params)

def opt():
    learning_rate = Real(low=5e-4, high=1e-1, prior='log-uniform',name='learning_rate')
    decay_rate = Real(low=0.1, high=0.98, name='decay_rate')
    step_size = Integer(low=10, high=100, name='step_size')
    seq_len = Categorical([4,8,16,32], name='seq_len')
    numBlocks = Categorical([2,3,4,5], name='numBlocks')
    numLayers = Categorical([2,3,4,5], name='numLayers')
    filters = Categorical([16,32,64,128,256], name='filters')

    default_parameters = [8e-3, 0.85, 75, 16, 3, 3, 64]
    dimensions = [learning_rate, decay_rate, step_size, seq_len, numBlocks, numLayers, filters]

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI', # Expected Improvement.
                                n_calls=12,
                                x0=default_parameters)

    optimized_params = search_result.x
    func_vals = search_result.func_vals
    x_iters = search_result.x_iters

    with open('opt_params.pkl', 'wb') as f:
        pickle.dump(optimized_params, f)

    df = pd.DataFrame(x_iters, columns=["ini_learning_rate", "decay_rate", "step_size", "seq_len", "numBlocks", "numLayers", "filters"])
    df.loc[:, ["Validation mse"]] = func_vals
    df.to_csv("result.csv")
    train_save_model(optimized_params)

def main():
    global target, image_1d
    target = []
    for file in os.listdir('tcnn_data'):
        if file[0] == 'f' and file[-1] == 't':
            target.append(parse(file))

    target = np.array(target)
    image_1d = target.reshape(-1,192)

    if args.train:
        train()

    if args.opt:
        opt()

    if args.eval:
        eval()

'''
structure
options

1. Train models on said hyperparameters
2. Optimise the model and return the best set of hyperparameters
3. Evaluate the model on the given set of hyperparameters
'''

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--train", type=int, default=0)
  parser.add_argument("--eval", type=int, default=1)
  parser.add_argument("--opt", type=int, default=0)
  parser.add_argument("--weights_path", type=str, default=None)
  parser.add_argument("--params_path", type=str, default=None)
  parser.add_argument("--ini_learning_rate", type=float, default=8e-3)
  parser.add_argument("--decay_rate", type=float, default=0.85)
  parser.add_argument("--step_size", type=int, default=75)
  parser.add_argument("--seq_len", type=int, default=16)
  parser.add_argument("--numBlocks", type=int, default=3)
  parser.add_argument("--numLayers", type=int, default=3)
  parser.add_argument("--filters", type=int, default=64)

  args = parser.parse_args()

  main()