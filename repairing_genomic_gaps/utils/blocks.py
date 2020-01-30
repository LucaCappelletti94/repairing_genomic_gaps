from tensorflow.keras.layers import Layer, BatchNormalization, Activation, Conv2D, Conv2DTranspose
from typing import List, Tuple

__all__ = ["encoder_blocks", "decoder_blocks"]


def autoencoder_blocks(
    x: Layer,
    layer: Layer,
    filters: List[int],
    kernels: List[Tuple[int, int]],
    strides: List[int]
) -> Layer:
    """Return a new stack of layers added in top of the given one.

    Parameters
    ----------
    x:Layer,
        The previous layer.
    layer:Layer,
        Class of the new layer to be instanziated.
        Can either be Conv2D or Conv2DTranspose
    filters:List[int],
        List of filters for the convolutional layer.
    kernels:List[Tuple[int, int]],
        List of kernel sizes for the convolutional layer.
    strides:List[int],
        List of strides for the convolutional layer.

    Returns
    -------
    Returns new top layer.
    """

    for f, k, s in zip(filters, kernels, strides):
        x = layer(
            filters=f,
            kernel_size=k,
            strides=s,
            padding='same'
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def encoder_blocks(
    x: Layer,
    filters: List[int],
    kernels: List[Tuple[int, int]],
    strides: List[int]
):
    """Return a new stack of Conv2D added in top of the given one.

    Parameters
    ----------
    x:Layer,
        The previous layer.
    filters:List[int],
        List of filters for the convolutional layer.
    kernels:List[Tuple[int, int]],
        List of kernel sizes for the convolutional layer.
    strides:List[int]
        List of strides for the convolutional layer.

    Returns
    -------
    Returns new top layer.
    """
    return autoencoder_blocks(x, Conv2D, filters, kernels, strides)


def decoder_blocks(
    x: Layer,
    filters: List[int],
    kernels: List[Tuple[int, int]],
    strides: List[int]
):
    """Return a new stack of Conv2DTranspose added in top of the given one.

    Parameters
    ----------
    x:Layer,
        The previous layer.
    filters:List[int],
        List of filters for the convolutional layer.
    kernels:List[Tuple[int, int]],
        List of kernel sizes for the convolutional layer.
    strides:List[int]
        List of strides for the convolutional layer.

    Returns
    -------
    Returns new top layer.
    """
    return autoencoder_blocks(x, Conv2DTranspose, filters, kernels, strides)
