import silence_tensorflow
from notipy_me import Notipy
from repairing_genomic_gaps import build_cnn_dataset, train_model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, add
from tensorflow.keras.layers import Flatten, Conv2DTranspose, Dropout
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import Sequence

if __name__ == "__main__":
    WINDOWS_SIZE = 64

    inputs = Input(shape=(WINDOWS_SIZE, 4))
    reshape = Reshape((WINDOWS_SIZE, 4, 1))(inputs)

    x = Conv2D(128, (20, 2), activation="relu")(reshape)
    x = BatchNormalization()(x)
    x = Conv2D(64, (10, 2), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (10, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (8, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (8, 1), activation="relu")(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    outputs = Dense(4, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"cnn_{WINDOWS_SIZE}")

    model.compile(
        optimizer=Nadam(),
        loss="categorical_crossentropy",
        metrics=[
            "categorical_accuracy",
        ]
    )
    model.summary()

    train, test = build_cnn_dataset(WINDOWS_SIZE, batch_size=128)
    model = train_model(model, train, test)