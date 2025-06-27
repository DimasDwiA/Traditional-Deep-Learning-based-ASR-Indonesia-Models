import os
import yaml
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
        
    def serializable_dict(self):
        data = self.__dict__.copy()

        # Change object with the serializable
        if isinstance(self.feature_extractor, MFCCExtractor):
            data['feature_extractor'] = {'type' : 'MFCCExtractor', 'frame_step' : self.feature_extractor.frame_step,
                                    'fft_length' : self.feature_extractor.fft_length, 'n_mfcc' :  self.feature_extractor.n_mfcc,
                                    'sr' : self.feature_extractor.sr}

        elif isinstance(self.feature_extractor, SpectrogramExtractor):
            data['feature_extractor'] = {'type' : 'SpectrogramExtractor', 'frame_step' : self.feature_extractor.frame_step,
                                    'fft_length' : self.feature_extractor.fft_length, 'n_mels' :  self.feature_extractor.n_mels,
                                    'sr' : self.feature_extractor.sr}

        # Learning rate
        if isinstance(self.learning_rate, tf.keras.optimizers.schedules.ExponentialDecay):
            data['learning_rate'] = {'name' : 'ExponentialDecay','learning_rate' : self.learning_rate.initial_learning_rate,
                                     'decay_steps' : self.learning_rate.decay_steps, 'decay_rate': self.learning_rate.decay_rate,
                                     'staircase': self.learning_rate.staircase}

        return data

    def save(self, path):
        with open(path, 'w') as f:
            yaml.safe_dump(self.serializable_dict(), f)

    def load(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        self.__dict__.update(data)

        # Rebuild feature_extractor
        fe = data.get('feature_extractor')
        if fe:
            if fe['type'] == 'MFCCExtractor':
                fe.pop('type')
                self.feature_extractor = MFCCExtractor(**fe)
            elif fe['type'] == 'SpectrogramExtractor':
                fe.pop('type')
                self.feature_extractor = SpectrogramExtractor(**fe)

        lr = data.get('learning_rate')
        if lr and lr.get('name') == 'ExponentialDecay':
            self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr["learning_rate"], 
                                                                                decay_steps=lr["decay_steps"], decay_rate=lr["decay_rate"],
                                                                                staircase=lr["staircase"])