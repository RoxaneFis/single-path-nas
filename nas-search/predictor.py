import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, initializers
from tensorflow.keras import backend as K
import numpy as np

class Predictor(tf.keras.Model):
    def __init__(self):
        super(Predictor, self).__init__()
        self.max_blocks = 37
        self.nb_nn_params = 7
        self.nb_hw_param = 12
        self._build()
    
    def _build(self):
        self.model_hw = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(self.nb_hw_param,)),
            layers.Dense(32, activation='linear')
        ])
        self.model_nn= tf.keras.Sequential([
                layers.Conv1D(128, (2), activation='relu', padding="same",input_shape=(self.max_blocks, self.nb_nn_params)),
                layers.Conv1D(32, (2), activation='relu', padding="same"),
                layers.Lambda( lambda x: K.sum(x, axis=1)),
        ])
        #self.predictor = tf.keras.Model(inputs=[self.model_nn.input, self.model_hw.input], outputs=[output])
        #self.Wsave =self.predictor.get_weights()
    def call(self, inputs, training=False):
        nn_pred = self.model_nn(inputs[0])
        hw_pred = self.model_hw(inputs[1])
        concat = tf.keras.layers.multiply([nn_pred, hw_pred])
        output = tf.keras.layers.Dense(units=16, activation='relu')(concat)
        output = tf.keras.layers.Dense(units=1, activation='relu')(output)

        return output

    # def predict(self, hw_array, nn_array):
    #     hw_array = hw_array.reshape((1, *hw_array.shape))
    #     nn_array = nn_array.reshape((1, *nn_array.shape))
    #     return self.predict([nn_array,hw_array])


def PredictorModel():
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


def PredictorModel_noWeights():
    max_blocks = 37
    nb_param = 7
    nb_hw_param = 12

    input_nn = Input(shape=(max_blocks, nb_param),  name='input_nn')
    input_hw = Input(shape=(nb_hw_param,),  name='input_hw')

    G_nn= np.ones((37,7))
    G_hw = np.ones((nb_hw_param,))

    output_nn = layers.Lambda(lambda x: x * G_nn)(input_nn)
    output_nn =tf.keras.layers.Lambda( lambda x: K.sum(x, axis=1))(output_nn)
    output_nn =tf.keras.layers.Lambda( lambda x: K.sum(x, axis=1))(output_nn)

    output_hw = layers.Lambda(lambda x: x * G_hw)(input_hw)
    output_hw =tf.keras.layers.Lambda( lambda x: K.sum(x, axis=1))(output_hw)
    concat = tf.keras.layers.multiply([output_nn, output_hw])

    model_no_weights= tf.keras.Model(inputs=[input_nn, input_hw], outputs=[concat])
    return model_no_weights

def PredictorModel_fewWeights():
    max_blocks = 37
    nb_param = 7
    nb_hw_param = 12

    input_nn = Input(shape=(max_blocks, nb_param),  name='input_nn')
    input_hw = Input(shape=(nb_hw_param,),  name='input_hw')

    G_nn= np.ones((37,7))
    G_hw = np.ones((nb_hw_param,))

    vec_nn = np.ones((7,1))

    output_nn = layers.Dense(1, kernel_initializer='zeros') (input_nn)
    #output_nn = layers.Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.01)) (input_nn)
    output_nn =tf.keras.layers.Lambda( lambda x: K.sum(x, axis=1))(output_nn)
    output_nn =tf.keras.layers.Lambda( lambda x: K.sum(x, axis=1))(output_nn)

    output_hw = layers.Lambda(lambda x: x * G_hw)(input_hw)
    output_hw =tf.keras.layers.Lambda( lambda x: K.sum(x, axis=1))(output_hw)

    concat = tf.keras.layers.multiply([output_nn, output_hw])
    concat = tf.keras.layers.Lambda( lambda x: x+6)(concat)

    model_few_weights= tf.keras.Model(inputs=[input_nn, input_hw], outputs=[concat])
    #model_path = "/Users/roxanefischer/Desktop/single_path_nas/single-path-nas/HAS/few_weights_0.794_error/few_weights"
    import pdb
    #pdb.set_trace()
    #model_few_weights.load_weights(model_path)
    return model_few_weights
