import silence_tensorflow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from multiprocessing import cpu_count
from notipy_me import Notipy
import pandas as pd
import os
from math import floor
from plot_keras_history import plot_history
from repairing_genomic_gaps import build_dataset_single, build_denoiser

from keras_synthetic_genome_sequence import GapSequence

from tensorflow.keras.layers import Input, Reshape, add
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.metrics import AUC

from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.utils import Sequence

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Conv2DTranspose

from repairing_genomic_gaps.utils import axis_softmax, axis_categorical

max_gap_size = 3
window_size = 500
batch_size = 1024
epochs = 1000


inputs = Input(shape=(window_size, 4))
reshape = Reshape((window_size, 4, 1))(inputs)

x = Conv2D(64, (10, 2), activation="relu")(reshape)
x = Conv2D(64, (10, 2), activation="relu")(x)
x = BatchNormalization()(x)
x = Conv2D(32, (10, 1), activation="relu")(x)
x = Conv2D(32, (10, 1), activation="relu")(x)
x = BatchNormalization()(x)
x = Conv2D(16, (10, 1), activation="relu")(x)
x = Conv2D(16, (10, 1), activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D((4, 1))(x)

x = Flatten()(x)

x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)
outputs = Dense(4, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs, name="cnn_500")

model.compile(
    optimizer=Nadam(),
    loss="categorical_crossentropy",
    metrics=[
        "categorical_accuracy",
    ]
)
model.summary()

saved_weights_path = "./models/{name}/best_weights_{name}.hdf5".format(name=model.name)

if os.path.exists(saved_weights_path):
    model.load_weights(saved_weights_path)
    print("Old Weights loaded from {}".format(saved_weights_path))

train, test = build_dataset_single(
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
            monitor='val_loss',
            min_delta=0.0001,
            patience=5,
            verbose=0,
            mode='min',
            restore_best_weights=True
        ),
        ModelCheckpoint(
            saved_weights_path,
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
).to_csv("./models/{name}/history_{name}.csv".format(name=model.name))
plot_history(history, path="./models/{name}/history_{name}.png".format(name=model.name))