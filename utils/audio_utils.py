import soundfile as sf
import numpy as np
import librosa
import random
import tensorflow as tf
from keras.saving import register_keras_serializable
from mltu.tensorflow.losses import CTCloss as OriginalCTCLoss

# Class to adjust CTCLoss
class CTCLossFixed(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.base_loss = OriginalCTCLoss(*args, **kwargs)

    def call(self, y_true, y_pred):
        loss = self.base_loss(y_true, y_pred)
        return tf.reduce_mean(loss)

# Function to save audio file correctly
def save_audio_file(audio_array, sample_rate, file_path):
    try:
        sf.write(file_path, audio_array, sample_rate)
    except Exception as e:
        print(f"Error saving audio file {file_path}: {e}")

# Function for normalize audio
def normalize_audio(audio_array):
    audio_array = audio_array / np.max(np.abs(audio_array))
    return audio_array

def augment_audio(audio, sr):
    # Time stretching
    if random.random() < 0.5:
        audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.1))  

    # Pitch shifting
    if random.random() < 0.5:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.randint(-2, 2))  

    # Add Gaussian noise
    if random.random() < 0.3:
        noise_amp = 0.005 * np.random.uniform() * np.amax(audio)
        audio = audio + noise_amp * np.random.normal(size=audio.shape)  

    audio = np.clip(audio, -1.0, 1.0)
    return audio

@register_keras_serializable()
def expand_dims_layer(x):
    return tf.expand_dims(x, axis=-1)