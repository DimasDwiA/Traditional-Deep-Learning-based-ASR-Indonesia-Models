import soundfile as sf
import numpy as np
import tensorflow as tf

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

def expand_dims_layer(x):
    return tf.expand_dims(x, axis=-1)