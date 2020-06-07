import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K

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


