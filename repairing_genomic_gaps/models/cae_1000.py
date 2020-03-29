from tensorflow.keras.models import Model
from .autoencoder_model import build_autoencoder


def cae_1000(
        verbose: bool = True,
        use_weighted : bool = False,
        _min : float = 1.0,
        _max : float = 10.0,
        return_encoder_and_decoder=False,
    ) -> Model:
    """Return autoencoder model for window sizes of 1000 nucleotides."""
    return build_autoencoder(
        input_shape=(1000, 4),
        latent_dim=200,
        filters=[64, 32, 16, 8],
        kernels=[(20, 4), (10, 4), (10, 2), (10, 1)],
        strides=[(5, 1), (5, 2), (2, 2), (1, 1)],
        verbose=verbose,
        use_weighted=use_weighted,
        _min=_min,
        _max=_max,
        return_encoder_and_decoder=return_encoder_and_decoder,
    )
