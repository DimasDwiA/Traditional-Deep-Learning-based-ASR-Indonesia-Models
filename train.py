import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.transformers import SpectrogramPadding, LabelIndexer, LabelPadding
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from models.model_factory import train_model_mfcc, train_model_spectrogram

def training(dataset_train, dataset_val, configs, mode):

    extractor = configs.feature_extractor
    
    if mode == 'mfcc':

        train_data_provider = DataProvider(
        dataset=dataset_train,
        skip_validation=True,
        batch_size=configs.batch_size,
        data_preprocessors=[extractor],
        transformers=[
            SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
            ]
        )

        val_data_provider = DataProvider(
        dataset=dataset_val,
        skip_validation=False,
        batch_size=configs.batch_size,
        data_preprocessors=[extractor],
        transformers=[
            SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
            ]
        )

        model_path = os.path.join(configs.model_path, 'model_100epochs.keras')
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model = load_model(model_path, custom_objects={
                "CTCloss" : CTCloss(),
                "CERMetrics" : CERMetric(vocabulary=configs.vocab),
                "WERMetrics" : WERMetric(vocabulary=configs.vocab)
            })
            initial_epoch = 100
        else:
            print("No saved model found, creating a new one.")
            model = train_model_mfcc(
                input_dim=configs.input_shape,
                output_dim=len(configs.vocab),
                dropout=0.3)
            initial_epoch = 0
        
    elif mode == 'spectrogram':

        train_data_provider = DataProvider(
        dataset=dataset_train,
        skip_validation=True,
        batch_size=configs.batch_size,
        data_preprocessors=[extractor],
        transformers=[
            SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
            ]
        )

        val_data_provider = DataProvider(
        dataset=dataset_val,
        skip_validation=False,
        batch_size=configs.batch_size,
        data_preprocessors=[extractor],
        transformers=[
            SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
            ]
        )

        model_path = os.path.join(configs.model_path, 'model_100epochs.keras')
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model = load_model(model_path, custom_objects={
                "CTCloss" : CTCloss(),
                "CERMetrics" : CERMetric(vocabulary=configs.vocab),
                "WERMetrics" : WERMetric(vocabulary=configs.vocab)
            })
            initial_epoch = 100
        else:
            print("No saved model found, creating a new one.")
            model = train_model_spectrogram(
                input_dim=configs.input_shape,
                output_dim=len(configs.vocab),
                dropout=0.3)
            initial_epoch = 0

    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if isinstance(configs.learning_rate, dict) and configs.learning_rate.get("name") == "ExponentialDecay":
        lr = configs.learning_rate
        configs.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr["learning_rate"],
            decay_steps=lr["decay_steps"],
            decay_rate=lr["decay_rate"],
            staircase=lr["staircase"]
        )
        
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab)
        ],
    run_eagerly=False)

    earlystopper = EarlyStopping(monitor="val_CER", patience=50, verbose=1, mode="min")
    checkpoint = ModelCheckpoint(f"{configs.model_path}/model.keras", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
    trainLogger = TrainLogger(configs.model_path)
    tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1, profile_batch='500,1000') # Profiling batch 500 sampai 1000
    reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.8, min_delta=1e-10, patience=50, verbose=1, mode="auto")
    model2onnx = Model2onnx(f"{configs.model_path}/model.keras")

    model.summary(line_length=110)

    history = model.fit(train_data_provider, validation_data=val_data_provider, epochs=configs.train_epochs,
                        initial_epoch=initial_epoch, callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx]
                       )
    
    print("Training Completed")
    
    return train_data_provider, val_data_provider, model, history