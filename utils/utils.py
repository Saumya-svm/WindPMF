import os
import numpy as np
from random import shuffle

def create_dir(dir_loc):
	if not os.path.exists(dir_loc):
		os.mkdir(dir_loc)

def parse(file):
    with open(f'tcnn_data/{file}') as f:
        x = []
        for i in range(12):
            x.append(f.readline().split())

    x = np.array(x)
    x = x.astype('float32')
    return x

def prepare_data(seq_len):
    target = []
    for file in os.listdir('tcnn_data'):
        if file[0] == 'f' and file[-1] == 't':
            target.append(parse(file))

    target = np.array(target)
    image_1d = target.reshape(-1, 192)
    X = []
    y = []
    for i in range(target.shape[0] - seq_len):
        X.append(image_1d[i:i + seq_len, :])
        y.append(target[i + 1:i + seq_len + 1, :, :])

    X = np.array(X)
    y = np.array(y)
    y = y.reshape(X.shape[0], seq_len, target.shape[1], target.shape[2])

    indices = [i for i in range(len(X))]
    shuffle(indices)

    X = X[indices]
    y = y[indices]
    train_size = int(X.shape[0] * 0.7)
    val_size = int(X.shape[0] * 0.9)
    trainX, trainY = X[:train_size], y[:train_size]
    valX, valY = X[train_size:val_size], y[train_size:val_size]
    testX, testY = X[val_size:], y[val_size:]
    return trainX, trainY, valX, valY, testX, testY