from tensorflow.keras.models import Model
from repairing_genomic_gaps import build_autoencoder


def cae_200() -> Model:
    """Return autoencoder model for window sizes of 200 nucleotides."""
    return build_autoencoder(
        input_shape=(200, 4),
        latent_dim=100,
        filters=[64, 32, 16],
        kernels=[(10, 4), (10, 4), (10, 2)],
        strides=[(2, 1), (5, 2), (2, 2)]
    )
