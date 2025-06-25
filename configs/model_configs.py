import os
import sys
import tensorflow as tf
from datetime import datetime 
from mltu.configs import BaseModelConfigs
from preprocessing.mfcc_extractor import MFCCExtractor
from preprocessing.spectrogram_extractor import SpectrogramExtractor

# Configuration
class ModelConfigs(BaseModelConfigs):
    def __init__(self, mode='mfcc'):
        super().__init__()
        self.model_path = os.path.join("D:/Project ASR/Model ASR Classic/saving_models/", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        os.makedirs(self.model_path, exist_ok=True)
        self.frame_length = 400
        self.frame_step = 512
        self.fft_length = 512
        self.sr = 16000
        self.n_mfcc = 40
        self.n_mels = 256

        self.train_dataset_path = None
        self.val_dataset_path = None

        self.vocab = "abcdefghijklmnopqrstuvwxyz0123456789'?!,. "

        self.input_shape = None
        self.max_text_length = None
        self.max_spectrogram_length = None

        self.batch_size = 64
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, 
                                                                            decay_steps=10000, 
                                                                            decay_rate=0.9)
        self.train_epochs = 300
        self.train_workers = 4

        self.mode = mode
        self.feature_extractor = self.get_extractor(mode)

    def get_extractor(self, mode):
        if mode == 'mfcc':
            return MFCCExtractor( 
                frame_step=self.frame_step,
                fft_length=self.fft_length,
                n_mfcc=self.n_mfcc,
                sr=self.sr  
            )
        
        elif mode == 'spectrogram':
            return SpectrogramExtractor(
                frame_step=self.frame_step,
                fft_length=self.fft_length,
                n_mels=self.n_mels,
                sr=self.sr
            )
        
        else:
            raise ValueError(f"Unknown mode: {mode}")