import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import argparse
import os
from utils import prepare_data
from model import prepare_model

def train_save_model(opt_params):
  print(opt_params)
  trainX, trainY, valX, valY, testX, testY = prepare_data(int(opt_params[3]))
  model, lr_sched = prepare_model(*opt_params)
  seq_len = opt_params[3]
  num_samples = trainX.shape[0]
  val_num_samples = valX.shape[0]
  model(trainX)
  model_fit = model.fit(tf.convert_to_tensor(trainX, dtype=tf.float32), tf.convert_to_tensor(trainY, dtype=tf.float32 ), epochs=800, batch_size=64, steps_per_epoch=num_samples//64,validation_data=(tf.convert_to_tensor(valX, dtype=tf.float32), tf.convert_to_tensor(valY, dtype=tf.float32)), callbacks=[lr_sched], validation_steps=val_num_samples//64)
  history = (model_fit)

  model.evaluate(trainX, trainY)
  model.save_weights("best_model")