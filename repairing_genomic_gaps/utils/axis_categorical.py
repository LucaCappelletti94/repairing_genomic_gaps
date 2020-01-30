import tensorflow as tf


def axis_categorical(target: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    """Define a categorical crossentropy that works on the one-hot encoded axis."""
    # This import is required for storing this lambda
    # with the model
    import tensorflow as tf
    return tf.keras.backend.categorical_crossentropy(
        target,
        output,
        axis=2
    )
