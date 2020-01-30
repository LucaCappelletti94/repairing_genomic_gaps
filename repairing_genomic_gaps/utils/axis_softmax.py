import tensorflow as tf


def axis_softmax(x: tf.Tensor) -> tf.Tensor:
    """Define a softmax that works on the one-hot encoded axis."""
    # This import is required for storing this lambda
    # with the model
    import tensorflow as tf
    return tf.nn.softmax(x, axis=2)
