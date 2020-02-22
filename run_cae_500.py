import silence_tensorflow
from notipy_me import Notipy
from repairing_genomic_gaps import build_autoenc_dataset, build_autoencoder, train_model

if __name__ == "__main__":
    WINDOWS_SIZE = 500

    model = build_autoencoder(
            input_shape=(WINDOWS_SIZE, 4),
            latent_dim=150,
            filters=[64, 32, 16, 8],
            kernels=[(10, 4), (10, 4), (10, 2), (10, 1)],
            strides=[(2, 1), (5, 2), (2, 2), (1, 1)]
        )

    with Notipy():
        train, test = build_autoenc_dataset(WINDOWS_SIZE, batch_size=512)
        model = train_model(model, train, test)
