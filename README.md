# Deep Learning based Indonesia ASR Models using MFCC and Spectrogram Features

This project implements an Automatic Speech Recognition (ASR) system in Python using TensorFlow and Keras. It supports training and evaluation of speech models based on **MFCC** or **Spectrogram** feature extraction methods using BiLSTM neural networks.

---

## 📦 Features

- Support for both **MFCC** and **Log Mel Spectrogram** feature extraction
- BiLSTM-based deep learning model for sequence classification
- CTC Loss for alignment-free transcription
- Custom evaluation metrics: **Character Error Rate (CER)** and **Word Error Rate (WER)**
- TensorBoard logging for training monitoring
- Automatic data loading and processing using HuggingFace datasets

---

## 🧪 Installation Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training Model
1. 📂 Prepare Dataset
   The dataset is downloaded and processed automatically using a script from main.py:
   - Audio data and transcripts are retrieved from Hugging Face (load_combined_dataset).
   - The dataset is split into training (train) and validation (validation) data.
   - Audio files and annotations are stored in data/raw/, and extracted into MFCC features or spectrograms in data/processed/.
     
   **Ensure that the connection to the Hugging Face API is successful before proceeding.**

2. ⚙️ Configure
   Model configuration is set through the ModelConfigs class:
   - Parameters include: sampling rate (sr), frame length, number of MFCC or Mel filters, batch size, and learning rate.
   - Configuration modes can be selected between:
       - “mfcc” for MFCC features
       - “spectrogram” for Log Mel Spectrogram features
    - Configurations are saved in a .yaml file for reproducibility.

3. ▶️ Run Training
   The model is trained by running:
   ```bash
    python main.py
    ```
   Or specify the mode explicitly:
   ```bash
    python main.py mfcc/spectrogram
   ```
4. 📊 Get Evaluation
   After training is complete, the model will be evaluated against the training and validation data:
   - Metrics displayed:
       - CER (Character Error Rate)
       - WER (Word Error Rate)
       - Loss
   - The evaluation results are printed to the console and the final model is saved for inference or deployment purposes. 

