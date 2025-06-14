from train import training
from configs.model_configs import ModelConfigs
from evaluate import evaluate
from preprocessing.mfcc_extractor import load_dataset_mfcc, process_mfcc_dataset
from preprocessing.spectrogram_extractor import load_dataset_spectrogram, process_spectrogram_dataset
from dataloader.data_loader import (login_huggingface, load_combined_dataset, process_dataset, 
                                    save_to_csv, organize_final_folders)

def main(mode='mfcc'):
    login_huggingface()

    # Proses training
    train_data = load_combined_dataset(split="train")
    train_results = process_dataset(train_data, "temp_audio")
    save_to_csv(train_results, "temp_audio/dataset_train.csv")
    organize_final_folders("temp_audio", "data/raw/train_audio", "data/raw/dataset_train.csv", "dataset_train.csv")

    # Proses validasi
    val_data = load_combined_dataset(split="validation")
    val_results = process_dataset(val_data, "temp_audio_val", is_validation=True)
    save_to_csv(val_results, "temp_audio_val/dataset_val.csv")
    organize_final_folders("temp_audio_val", "data/raw/val_audio", "data/raw/dataset_val.csv", "dataset_val.csv")

    if mode == 'mfcc':
        configs = ModelConfigs(mode=mode)

        train_dataset = load_dataset_mfcc('D:/Project ASR/Model ASR Classic/data/raw/dataset_train.csv', 'D:/Project ASR/Model ASR Classic/data/raw/train_audio')
        val_dataset = load_dataset_mfcc('D:/Project ASR/Model ASR Classic/data/raw/dataset_val.csv', 'D:/Project ASR/Model ASR Classic/data/raw/val_audio')

        max_text_train, max_mfcc_train = process_mfcc_dataset(train_dataset, configs, "audio_train")
        max_text_val, max_mfcc_val = process_mfcc_dataset(val_dataset, configs, "audio_val")

        configs.max_text_length = max(max_text_train, max_text_val)
        configs.max_spectrogram_length = max(max_mfcc_train, max_mfcc_val)

        configs.input_shape = [configs.max_spectrogram_length, configs.n_mfcc]

        configs.save("D:/Project ASR/Model ASR Classic/data/processed/mfcc/configs.yaml")

    elif mode == 'spectrogram':
        configs = ModelConfigs(mode=mode)

        train_dataset = load_dataset_spectrogram('D:/Project ASR/Model ASR Classic/data/raw/dataset_train.csv', 'D:/Project ASR/Model ASR Classic/data/raw/train_audio')
        val_dataset = load_dataset_spectrogram('D:/Project ASR/Model ASR Classic/data/raw/dataset_train.csv', 'D:/Project ASR/Model ASR Classic/data/raw/train_audio')

        max_text_train, max_spec_train = process_spectrogram_dataset(train_dataset, configs, "audio_train")
        max_text_val, max_spec_val = process_spectrogram_dataset(val_dataset, configs, "audio_val")

        configs.max_text_length = max(max_text_train, max_text_val)
        configs.max_spectrogram_length = max(max_spec_train, max_spec_val)

        configs.input_shape = [configs.max_spectrogram_length, configs.n_mels]

        configs.save("D:/Project ASR/Model ASR Classic/data/processed/spectrogram/configs.yaml")

    train_data_provider, val_data_provider, model, history = training(train_dataset, val_dataset, configs, mode)

    evaluate(train_data_provider, val_data_provider, model, configs)

if __name__ == '__main__':
    main('spectrogram')
