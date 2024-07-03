import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, UpSampling2D, TimeDistributed, Reshape
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from utils import step_decay_schedule

class ResBlock(tf.keras.layers.Layer):
  def __init__(self, num_layers=3, dilations=[1,2,4], dropoutProb=0.25, filters=64, padding='causal'):
    super(ResBlock, self).__init__()
    self.weightNorm = tfa.layers.WeightNormalization
    self.conv = Conv1D
    self.relu = tf.keras.layers.ReLU
    self.dropout = tf.keras.layers.Dropout(dropoutProb)
    self.training = True
    self.filters = filters
    self.padding = padding
    self.activation = 'relu'
    self.layers = []
    # assert whether the number of layers and the len of dilations match or not
    assert num_layers == len(dilations), "Dilations undefined for some layers"

    self.num_layers = num_layers
    self.dilations = dilations
    self.recon_conv = self.conv(self.filters,1)

  def make_layers(self):
    for i in range(self.num_layers):
      self.layers.extend([self.weightNorm(self.conv(self.filters,2, dilation_rate=self.dilations[i],
                                                    padding=self.padding, activation=self.activation)),
                          self.relu(),
                          self.dropout])

  def call(self, input):
    assert self.layers, "Call make_layers function before forward pass"

    # input = tf.keras.layers.Input(shape=(seq_len,192))
    x = input
    for layer in self.layers:
      if isinstance(layer, tf.keras.layers.Dropout):
        x = layer(x)
        continue

      x = layer(x)

    input1 = self.recon_conv(input)
    input1 = self.relu()(tf.keras.layers.add([x, input1]))
    x = input1
    return x

class TCNN(tf.keras.Model):
  def __init__(self, num_blocks=3, num_layers=3, dilations=[1,2,4], dropoutProb=0.25, filters=64, padding='causal'):
    super().__init__()
    self.blocks = []
    self.num_blocks = num_blocks
    dilations = [2**i for i in range(num_layers)]
    self.resBlockInput = ResBlock(num_layers, dilations, dropoutProb, filters, padding)
    self.resBlockInput.make_layers()

    self.ffn = Dense(192, activation='softmax')
    self.num_layers = num_layers
    self.dilations = dilations
    self.dropoutProb = dropoutProb
    self.filters = filters
    self.padding = padding

  def make_model(self):
    self.blocks.append(self.resBlockInput)
    for i in range(1,self.num_blocks):
      temp = ResBlock(self.num_layers, self.dilations, self.dropoutProb, self.filters, self.padding)
      temp.make_layers()
      self.blocks.append(temp)

    # self.model = tf.keras.Sequential(self.blocks)

  def call(self, x):
    # input = tf.keras.layers.Input(shape=(seq_len,192))
    # x = input
    batch_size = x.shape[0]
    for block in self.blocks:
      x = block(x)
      # print("Hello")
    # print("single pass")
    x = self.ffn(x)
    # x = self.flatten(x)
    seq_len = x.shape[1]
    output = tf.keras.layers.Reshape((seq_len,12,16))(x)
    # print("returning output", type(output))
    return output

def prepare_model(learning_rate, decay_rate, step_size, seq_len, numBlocks, numLayers, filters):
  seq_len, numBlocks, filters = int(seq_len), int(numBlocks), int(filters)
  model = TCNN(num_blocks=numBlocks, filters=filters, num_layers=numLayers)
  opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(loss='mse',
              optimizer='adam',
              metrics=[tf.keras.metrics.MeanSquaredError()])
  model.make_model()
  lr_sched = step_decay_schedule(initial_lr=learning_rate, decay_factor=decay_rate, step_size=step_size)

  return model, lr_sched