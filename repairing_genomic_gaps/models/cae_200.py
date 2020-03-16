from tensorflow.keras.models import Model
from .autoencoder_model import build_autoencoder


def cae_200(
        verbose: bool = True,
        use_weighted : bool = False,
        _min : float = 1.0,
        _max : float = 10.0,
        ) -> Model:
    """Return autoencoder model for window sizes of 200 nucleotides."""
    return build_autoencoder(
        input_shape=(200, 4),
        latent_dim=100,
        filters=[64, 32, 16],
        kernels=[(10, 4), (10, 4), (10, 2)],
        strides=[(2, 1), (5, 2), (2, 2)],
        verbose=verbose,
        use_weighted=use_weighted,
        _min=_min,
        _max=_max,
    )
