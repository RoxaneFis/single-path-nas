import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, initializers
from tensorflow.keras import backend as K
import numpy as np


def PredictorConv1D():
    max_blocks = 37
    nb_param = 7
    nb_hw_param = 12
    model_nn= tf.keras.Sequential([
        layers.Conv1D(128, (2), activation='relu', padding="same",input_shape=(max_blocks, nb_param)),
        layers.Conv1D(32, (2), activation='relu', padding="same"),
        layers.Lambda( lambda x: K.sum(x, axis=1)),
    ])
    model_hw = tf.keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(nb_hw_param,)),
        layers.Dense(32, activation='linear')
    ])
    concat = tf.keras.layers.multiply([model_nn.output, model_hw.output])
    output = tf.keras.layers.Dense(units=16, activation='relu')(concat)
    output = tf.keras.layers.Dense(units=1, activation='relu')(output)
    full_model = tf.keras.Model(inputs=[model_nn.input, model_hw.input], outputs=[output])
    return full_model




def PredictorRNN():
    max_blocks = 37
    nb_param = 7
    nb_hw_param = 12
    last_layer =128
    output_shape = 1
    input_nn = Input(shape=(max_blocks, nb_param), dtype='float32', name='input_nn')
    input_hw = Input(shape=(nb_hw_param,), dtype='float32', name='input_hw')

    output_nn = layers.LSTM(128, return_sequences=True)(input_nn)
    output_nn=layers.Dense(last_layer,  activation='relu')(output_nn)
    output_nn =tf.keras.layers.Lambda( lambda x: K.sum(x, axis=1))(output_nn)

   # output_hw = Recover()(input_hw)
    output_hw = layers.Dense(128, activation='relu')(input_hw)
    #output_hw = layers.Dense(128, activation='relu')(output_hw)
    output_hw = layers.Dense(last_layer, activation='linear')(output_hw)

    concat = tf.keras.layers.multiply([output_nn, output_hw])
    concat = tf.keras.layers.Concatenate()([concat,output_nn, output_hw])

    output = tf.keras.layers.Dense(units=32, activation='relu')(concat)
    output = tf.keras.layers.Dense(units=32, activation='relu')(output)
    output = tf.keras.layers.Dense(units=output_shape, activation='relu')(output)
    full_model = tf.keras.Model(inputs=[input_nn, input_hw], outputs=[output])
    return full_model




