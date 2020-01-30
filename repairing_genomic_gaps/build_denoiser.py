from .models import build_autoencoder


def build_denoiser(
    window_size: int
):
    return build_autoencoder(
        input_shape=(window_size, 4),
        latent_dim=100,
        filters=[128, 64, 32],
        kernels=[(12, 4), (6, 4), (3, 2)],
        strides=[(1, 1), (2, 2), (2, 2)]
    )
