from repairing_genomic_gaps import build_dataset, build_denoiser
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_tqdm import TQDMCallback
from multiprocessing import cpu_count
from notipy_me import Notipy
import pandas as pd
from plot_keras_history import plot_history

max_gap_size = 100
window_size = 500
batch_size = 256
batch_number = 10000
epochs = 1000

with Notipy():
    train, test = build_dataset(
        assembly="hg19",
        training_chromosomes=[
            "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
            "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr19",
            "chr20", "chr21", "chr22", "chrX", "chrY"
        ],
        max_training_samples=batch_size*batch_number,
        testing_chromosomes=[
            "chr17",
            "chr18",
            "chrM",
        ],
        max_testing_samples=batch_size*batch_number,
        max_gap_size=max_gap_size,
        window_size=window_size,
        gaps_threshold=0.4,
        batch_size=batch_size,
        seed=42
    )

    model = build_denoiser(window_size)

    history = model.fit_generator(
        train,
        steps_per_epoch=train.steps_per_epoch,
        epochs=epochs,
        shuffle=True,
        verbose=0,
        callbacks=[
            TQDMCallback(),
            EarlyStopping(
                monitor='loss',
                min_delta=0.0001,
                patience=10,
                verbose=0,
                mode='min',
                restore_best_weights=True
            ),
            ModelCheckpoint(
                "best_model.hdf5",
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                period=1
            )
        ],
        validation_data=test,
        validation_steps=test.steps_per_epoch,
        workers=cpu_count()//2,
        use_multiprocessing=True
    ).history
    pd.DataFrame(
        history
    ).to_csv("history.csv")
    plot_history(history, path="standard.png")
