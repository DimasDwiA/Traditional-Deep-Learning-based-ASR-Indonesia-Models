import os
import sys
import json
import pandas as pd
import shutil
from tqdm import tqdm
import subprocess
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
from dotenv import load_dotenv
from utils.audio_utils import save_audio_file, normalize_audio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath('D:/Project ASR/Model ASR Classic')

def login_huggingface():
    load_dotenv()
    login(token=os.getenv("TOKEN_HUGGINGFACE"))

def load_combined_dataset(split="train"):
    data_13 = load_dataset("mozilla-foundation/common_voice_13_0", "id", split=split, trust_remote_code=True)
    data_15 = load_dataset("mozilla-foundation/common_voice_15_0", "id", split=split, trust_remote_code=True)
    data_16 = load_dataset("mozilla-foundation/common_voice_16_0", "id", split=split, trust_remote_code=True)
    return concatenate_datasets([data_13, data_15, data_16])

def process_dataset(dataset, temp_dir, is_validation=False):
    os.makedirs(temp_dir, exist_ok=True)
    results = []

    for i, example in tqdm(enumerate(dataset), desc=f"Processing {'validation' if is_validation else 'training'} dataset"):
        try:
            sentence = example["sentence"].strip().lower()
            if not sentence:
                continue

            audio_array = example["audio"]["array"]
            sample_rate = example["audio"]["sampling_rate"]
            if audio_array is None or len(audio_array) == 0:
                continue

            audio_array = normalize_audio(audio_array)

            filename = f"audio_val_{i}.wav" if is_validation else f"audio_{i}.wav"
            audio_path = os.path.join(temp_dir, filename)
            save_audio_file(audio_array, sample_rate, audio_path)

            results.append({"file": filename, "sentence": sentence})
        except Exception as e:
            print(f"Skipping index {i} due to error: {e}")
            continue

    return results

def save_to_csv(data, output_path):
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

def organize_final_folders(temp_dir, final_audio_dir, final_csv_path, temp_csv_name):
    os.makedirs(final_audio_dir, exist_ok=True)
    shutil.move(os.path.join(temp_dir, temp_csv_name), final_csv_path)
    for f in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, f), os.path.join(final_audio_dir, f))

def prepare_kaggle_dataset():
    os.makedirs("kaggle-dataset-asr", exist_ok=True)
    metadata = {
        "title": "Dataset ASR Common Voice",
        "id": "Dimzzzz/kaggle-dataset-asr",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open("kaggle-dataset-asr/dataset-metadata.json", "w") as f:
        json.dump(metadata, f)

    shutil.copy("data/raw/dataset_train.csv", "kaggle-dataset-asr/dataset_train.csv")
    shutil.copy("data/raw/dataset_val.csv", "kaggle-dataset-asr/dataset_val.csv")
    shutil.copytree("data/raw/train_audio", "kaggle-dataset-asr/train_audio", dirs_exist_ok=True)
    shutil.copytree("data/raw/val_audio", "kaggle-dataset-asr/val_audio", dirs_exist_ok=True)

def upload_kaggle_dataset():
    shutil.make_archive("kaggle-dataset-asr", "zip", "kaggle-dataset-asr")
    os.makedirs("kaggle_upload", exist_ok=True)
    shutil.move("kaggle-dataset-asr.zip", "kaggle_upload/kaggle-dataset-asr.zip")
    shutil.copy("kaggle-dataset-asr/dataset-metadata.json", "kaggle_upload/dataset-metadata.json")
    subprocess.run(['kaggle', 'datasets', 'create', '-p', 'kaggle_upload'])
