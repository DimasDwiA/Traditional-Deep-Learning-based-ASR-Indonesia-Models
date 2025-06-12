import os
import sys
import csv
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath('D:/Project ASR')


# MFCC extraction in DataProvider
class MFCCExtractor:
    def __init__(self, frame_step, fft_length, n_mfcc, sr):
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.n_mfcc = n_mfcc
        self.sr = sr

    def __call__(self, data, annotation):
        file_path = data
        return extract_mfcc(
            file_path, 
            self.frame_step, 
            self.fft_length, 
            self.n_mfcc,
            self.sr
        ), annotation
    
# Function to extract feature with mfcc 
def extract_mfcc(audio, frame_step, fft_length, n_mfcc, sr):
    # Load audio
    if isinstance (audio, str):
        audio, _ = librosa.load(audio, sr=sr)
    
    # Pre-emphasis filter
    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=fft_length,
        hop_length=frame_step,
        window='hamming',
        center=True,
        power=2.0
    )
    
    # Normalize MFCC features
    mfccs = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-9)
    
    return mfccs.T.astype(np.float32)

def load_dataset_mfcc(csv_path, audio_base_path):
    df = pd.read_csv(csv_path)
    return [(os.path.join(audio_base_path, row["file"]), row["sentence"]) for _, row in df.iterrows()]


def process_mfcc_dataset(dataset, configs, save_subdir="audio_train"):
    output_dir = "data/processed/mfcc"
    os.makedirs(os.path.join(output_dir, save_subdir), exist_ok=True)

    max_text_length = 0
    max_mfcc_length = 0
    processed_mfcc = []

    for file_path, label in tqdm(dataset, desc=f"Saving {save_subdir} MFCC"):
        label_clean = [c for c in label if c in configs.vocab]
        if len(label_clean) == 0:
            continue

        try:
            audio_array, _ = sf.read(file_path, dtype="float32")
            # Extract MFCC from audio
            mfcc = extract_mfcc(audio_array, configs.frame_step, configs.fft_length, configs.n_mfcc, configs.sr)

            max_text_length = max(max_text_length, len(label_clean))
            max_mfcc_length = max(max_mfcc_length, mfcc.shape[0])

            processed_mfcc.append((file_path, label_clean))

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if len(processed_mfcc) == 0:
        print("⚠️ Tidak ada MFCC yang berhasil diproses.")

    processed_csv_mfcc  = os.path.join(output_dir, f"{save_subdir}_mfcc_processed.csv")
    with open(processed_csv_mfcc, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'sentence'])
        writer.writerows(processed_mfcc)    

    return max_text_length, max_mfcc_length
