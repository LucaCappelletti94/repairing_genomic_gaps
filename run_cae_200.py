import silence_tensorflow

from repairing_genomic_gaps import build_autoenc_dataset, build_autoencoder, train_model

WINDOWS_SIZE = 200

model = build_autoencoder(
        input_shape=(WINDOWS_SIZE, 4),
        latent_dim=100,
        filters=[64, 32, 16],
        kernels=[(10, 4), (10, 4), (10, 2)],
        strides=[(2, 1), (5, 2), (2, 2)]
    )

train, test = build_autoenc_dataset(WINDOWS_SIZE)
model = train_model(model, train, test)
