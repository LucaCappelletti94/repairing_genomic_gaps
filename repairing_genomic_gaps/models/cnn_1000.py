
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization

def cnn_1000():
    inputs = Input(shape=(1000, 4))
    reshape = Reshape((1000, 4, 1))(inputs)

    x = Conv2D(32, (10, 2), activation="relu")(reshape)
    x = Conv2D(64, (10, 2), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (10, 1), activation="relu")(x)
    x = Conv2D(32, (10, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((4, 1))(x)
    x = Conv2D(32, (10, 1), activation="relu")(x)
    x = Conv2D(64, (10, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (10, 1), activation="relu")(x)
    x = Conv2D(16, (10, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((4, 1))(x)

    x = Flatten()(x)

    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(4, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="cnn_1000")
    return model