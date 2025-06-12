import tensorflow as tf
from keras import Model
from utils.audio_utils import expand_dims_layer
from keras.layers import LSTM, Input, Bidirectional, BatchNormalization, Activation, Add, Dropout, Dense, Conv1D, Conv2D, Reshape, Lambda, MultiHeadAttention, LayerNormalization

# Function model ASR for tarining data mfcc
def train_model_mfcc(input_dim, output_dim, activation="leaky_relu", dropout=0.3):

    inputs = Input(shape=input_dim, name="input", dtype=tf.float32)

    # BiLSTM layers 1
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    # BiLSTM layers 2
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    # BiLSTM layers 3
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    # Dense layer
    x = Dense(256)(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    # Classification layer
    output = Dense(output_dim + 1, activation="softmax", dtype=tf.float32)(x)

    model = Model(inputs=inputs, outputs=output)
    return model

# Function model ASR for tarining data spectrogram
def train_model_spectrogram(input_dim, output_dim, activation="leaky_relu", dropout=0.3):
    
    inputs = Input(shape=input_dim, name="input", dtype=tf.float32)

    # expand dims to add channel dimension
    input = Lambda(expand_dims_layer, output_shape=lambda s: s + (1,), name="expand_dims")(inputs)

    # Convolution layer 1
    x = Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    # Convolution layer 2
    x = Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    
    # Reshape the resulted volume to feed the RNNs layers
    x = Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # BiLSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    # Dense layer
    x = Dense(256)(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    # Classification layer
    output = Dense(output_dim + 1, activation="softmax", dtype=tf.float32)(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model