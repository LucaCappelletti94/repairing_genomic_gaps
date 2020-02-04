from .models import build_autoencoder


def build_denoiser(
    window_size: int
):
    return build_autoencoder(
        input_shape=(window_size, 4),
        latent_dim=100,
        filters=[64, 32, 16, 8],
        kernels=[(9, 4), (6, 4), (3, 2), (3, 1)],
        strides=[(2, 1), (2, 2), (2, 2), (1, 1)]
    )
