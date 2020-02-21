import silence_tensorflow
from notipy_me import Notipy
from repairing_genomic_gaps import build_autoenc_dataset, build_autoencoder, train_model

WINDOWS_SIZE = 1000

model = build_autoencoder(
        input_shape=(WINDOWS_SIZE, 4),
        latent_dim=200,
        filters=[64, 32, 16, 8],
        kernels=[(20, 4), (10, 4), (10, 2), (10, 1)],
        strides=[(5, 1), (5, 2), (2, 2), (1, 1)]
    )

with Notipy():
    train, test = build_autoenc_dataset(WINDOWS_SIZE)
    model = train_model(model, train, test)
