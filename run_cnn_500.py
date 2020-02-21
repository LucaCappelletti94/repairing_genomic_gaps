import silence_tensorflow
from repairing_genomic_gaps import build_cnn_dataset, train_model
from notipy_me import Notipy

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, add
from tensorflow.keras.layers import Flatten, Conv2DTranspose
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import Sequence
WINDOWS_SIZE = 500

inputs = Input(shape=(WINDOWS_SIZE, 4))
reshape = Reshape((WINDOWS_SIZE, 4, 1))(inputs)

x = Conv2D(32, (10, 2), activation="relu")(reshape)
x = Conv2D(64, (10, 2), activation="relu")(x)
x = BatchNormalization()(x)
x = Conv2D(64, (10, 1), activation="relu")(x)
x = Conv2D(32, (10, 1), activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D((2, 1))(x)
x = Conv2D(32, (10, 1), activation="relu")(x)
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

with Notipy():
    train, test = build_cnn_dataset(WINDOWS_SIZE)
    model = train_model(model, train, test)