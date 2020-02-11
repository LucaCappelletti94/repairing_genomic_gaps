import silence_tensorflow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from multiprocessing import cpu_count
from notipy_me import Notipy
import pandas as pd
from plot_keras_history import plot_history
from repairing_genomic_gaps import build_dataset, build_denoiser

max_gap_size = 3
window_size = 200
batch_size = 250
epochs = 1000

with Notipy():

    model = build_denoiser(window_size)
    
    train, test = build_dataset(
        assembly="hg19",
        training_chromosomes=[
            "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
            "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr19",
            "chr20", "chr21", "chr22", "chrX", "chrY"
        ],
        testing_chromosomes=[
            "chr17",
            "chr18",
            "chrM",
        ],
        max_gap_size=max_gap_size,
        window_size=window_size,
        gaps_threshold=0.4,
        batch_size=batch_size,
        seed=42
    )

    history = model.fit_generator(
        train,
        steps_per_epoch=train.steps_per_epoch/15,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[
            EarlyStopping(
                monitor='loss',
                min_delta=0.0001,
                patience=10,
                verbose=0,
                mode='min',
                restore_best_weights=True
            ),
            ModelCheckpoint(
                "best_small_model.hdf5",
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='min'
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
