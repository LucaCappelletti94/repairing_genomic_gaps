import tensorflow as tf

def weighted_axis_categorical(_min : float, _max : float):
    _min, _max = float(_min), float(_max)
    def axis_categorical(target: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        """Define a categorical crossentropy that works on the one-hot encoded axis."""
        # This import is required for storing this lambda
        # with the model
        dim = tf.shape(target)[1]
        upper = tf.math.ceil(dim / 2)
        lower = tf.math.floor(dim / 2)
        upper, lower = tf.cast(upper, tf.int32), tf.cast(lower, tf.int32) 
        weights = tf.concat(
            [
                tf.linspace(_min, _max, upper),
                tf.linspace(_max, _min, lower),
            ],
            axis=0
        )
        values = tf.keras.backend.categorical_crossentropy(
            target,
            output,
            axis=2
        )
        return tf.linalg.matvec(values, weights)
    return axis_categorical