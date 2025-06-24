import os
import sys
import csv
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Spectrogram extraction in DataProvider
class SpectrogramExtractor:
    def __init__(self, frame_step, fft_length, n_mels, sr):
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.n_mels = n_mels
        self.sr = sr

    def __call__(self, data, annotation):
        file_path = data
        return extract_log_mel_spectrogram(
            file_path,  
            self.frame_step, 
            self.fft_length, 
            self.n_mels,
            self.sr
        ), annotation
    
# Function to extract feature with spectrogram 
def extract_log_mel_spectrogram(audio, frame_step, fft_length, n_mels, sr):
    # Load audio
    if isinstance (audio, str):
        audio, _ = librosa.load(audio, sr=sr)

    # Pre-emphasis filter
    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    mel_spec = librosa.feature.melspectrogram(y=audio, n_mels=n_mels, n_fft=fft_length, hop_length=frame_step,
                                              power=2.0, window='hamming')
    
    # Normalize mel spec
    log_mel_spec = librosa.power_to_db(mel_spec)
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec, axis=1, keepdims=True)) / (np.std(log_mel_spec, axis=1, keepdims=True) + 1e-9)
    
    return log_mel_spec.T.astype(np.float32)

def load_dataset_spectrogram(csv_path, audio_base_path):
    df = pd.read_csv(csv_path)
    return [(os.path.join(audio_base_path, row["file"]), row["sentence"]) for _, row in df.iterrows()]

def process_spectrogram_dataset(dataset, configs, save_subdir="audio_train"):
    output_dir = "data/processed/spectrogram"
    os.makedirs(os.path.join(output_dir, save_subdir), exist_ok=True)

    max_text_length = 0
    max_mel_length = 0
    processed_spec = []

    for file_path, label in tqdm(dataset, desc=f"Saving {save_subdir} Spectrogram"):
        label_clean = [c for c in label if c in configs.vocab]
        if len(label_clean) == 0:
            continue

        try:
            audio_array, _ = sf.read(file_path, dtype="float32")
            # Extract log mel spectogram from audio
            log_mel_spectogram = extract_log_mel_spectrogram(audio_array, configs.frame_step, 
                                                             configs.fft_length, configs.n_mels, configs.sr)

            max_text_length = max(max_text_length, len(label_clean))
            max_mel_length = max(max_mel_length, log_mel_spectogram.shape[0])

            processed_spec.append((file_path, label_clean))

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if len(processed_spec) == 0:
        print("⚠️ Tidak ada Spectogram yang berhasil diproses.")  

    processed_csv_spec = os.path.join(output_dir, f"{save_subdir}_spec_processed.csv")
    with open(processed_csv_spec, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'sentence'])
        writer.writerows(processed_spec) 

    return max_text_length, max_mel_length
