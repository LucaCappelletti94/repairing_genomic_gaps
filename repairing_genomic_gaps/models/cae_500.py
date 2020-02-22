from tensorflow.keras.models import Model
from repairing_genomic_gaps import build_autoencoder


def cae_500() -> Model:
    """Return autoencoder model for window sizes of 500 nucleotides."""
    return build_autoencoder(
        input_shape=(500, 4),
        latent_dim=150,
        filters=[64, 32, 16, 8],
        kernels=[(10, 4), (10, 4), (10, 2), (10, 1)],
        strides=[(2, 1), (5, 2), (2, 2), (1, 1)]
    )
