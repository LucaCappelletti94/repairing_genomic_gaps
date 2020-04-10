from typing import Tuple, List
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Conv2DTranspose
from ..utils import axis_softmax, axis_categorical, weighted_axis_categorical
from .autoencoder_blocks import encoder_blocks, decoder_blocks


def build_encoder(
    input_shape: Tuple,
    latent_dim: int,
    filters: List[int],
    kernels: List[Tuple[int, int]],
    strides: List[int],
) -> Tuple[Input, Tuple, Model]:
    """Return Tuple containing inputs, last convolutional layer shape and encoder model.

    Parameters
    ----------
    input_shape:Tuple,
        Tuple with input shape of the model.
    latent_dim:int,
        Size of the latent vector.
    filters:List[int],
        List of filters for the convolutional layer.
    kernels:List[Tuple[int, int]],
        List of kernel sizes for the convolutional layer.
    strides:List[int]
        List of strides for the convolutional layer.

    """
    inputs = Input(shape=input_shape, name='encoder_input')
    reshape = Reshape((*input_shape, 1))(inputs)
    x = encoder_blocks(reshape, filters, kernels, strides)
    # Shape info needed to build Decoder Model
    shape = tf.keras.backend.int_shape(x)[1:]
    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)
    # Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    return inputs, shape, encoder


def build_decoder(
    latent_dim: int,
    input_shape: Tuple,
    encoder_shape: Tuple,
    filters: List[int],
    kernels: List[Tuple[int, int]],
    strides: List[int]
) -> Model:
    """Return decoder model.

    Parameters
    ----------
    latent_dim:int,
        Size of the latent vector.
    encoder_shape:Tuple,
        Output shape of the last convolutional layer
        of the encoder model.
    filters:List[int],
        List of filters for the convolutional layer.
    kernels:List[Tuple[int, int]],
        List of kernel sizes for the convolutional layer.
    strides:List[int]
        List of strides for the convolutional layer.

    """
    decoder_input = Input(
        shape=(latent_dim,),
        name='decoder_input'
    )
    x = Dense(np.prod(encoder_shape))(decoder_input)
    x = Reshape(encoder_shape)(x)
    x = decoder_blocks(
        x,
        reversed(filters),
        reversed(kernels),
        reversed(strides)
    )
    decoder_output = Conv2DTranspose(
        filters=1,
        kernel_size=kernels[0],
        activation=axis_softmax,
        padding='same',
        name='decoder_output'
    )(x)
    reshape = Reshape(input_shape)(decoder_output)

    # Instantiate Decoder Model
    return Model(
        decoder_input,
        reshape,
        name='decoder'
    )


def build_autoencoder(
    input_shape: Tuple,
    latent_dim: int,
    filters: List[int],
    kernels: List[Tuple[int, int]],
    strides: List[int],
    verbose: bool,
    use_weighted : bool = False,
    _min : float = 1.0,
    _max : float = 10.0,
    return_encoder_and_decoder=False,
):
    inputs, encoder_shape, encoder = build_encoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        filters=filters,
        kernels=kernels,
        strides=strides,
    )

    decoder = build_decoder(
        latent_dim=latent_dim,
        input_shape=input_shape,
        encoder_shape=encoder_shape,
        filters=filters,
        strides=strides,
        kernels=kernels
    )

    # Autoencoder = Encoder + Decoder
    # Instantiate Autoencoder Model
    autoencoder = Model(
        inputs,
        decoder(encoder(inputs)),
        name='cae_{}'.format(input_shape[0])
    )


    if use_weighted:
        loss = weighted_axis_categorical(_min, _max)
    else:
        loss = axis_categorical

    autoencoder.compile(
        loss=loss,
        optimizer='nadam'
    )

    if verbose:
        encoder.summary()
        decoder.summary()
        autoencoder.summary()
        
    if return_encoder_and_decoder:
        return autoencoder, encoder, decoder

    return autoencoder
