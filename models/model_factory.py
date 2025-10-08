import tensorflow as tf
from tensorflow.keras import regularizers
from utils.audio_utils import expand_dims_layer
from keras import Model
from keras.layers import (LSTM, Input, Bidirectional, BatchNormalization, Dropout, Dense, Conv1D, Conv2D, Reshape, Lambda, 
                          MaxPooling2D, Reshape, Attention, Add, LeakyReLU, Activation)

# Function model ASR for tarining data mfcc
def train_model_mfcc(input_dim, output_dim, dropout=0.3, l2_reg=1e-4):

    inputs = Input(shape=input_dim, name="input", dtype=tf.float32)

    # BiLSTM layers 1
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    # BiLSTM layers 2
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    # BiLSTM layers 3
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    attn = Attention()([x, x])
    x = Add()([x, attn])

    # Dense layer
    x = Dense(256, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LeakyReLU()(x)
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
    x = Conv2D(64, (3,3), padding="same", use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    # Convolution layer 2
    x = Conv2D(64, (3,3), padding='same', activation=activation)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
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

    # Dense layer
    x = Dense(256)(x)
    x = Activation(activation=activation)(x)
    x = Dropout(dropout)(x)

    # Classification layer
    output = Dense(output_dim + 1, activation="softmax", dtype=tf.float32)(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model