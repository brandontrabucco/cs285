"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def dense(
    input_size,
    output_size,
    hidden_size=400,
    num_hidden_layers=1,
):
    visible = tf.keras.layers.Input(shape=(input_size,))
    hidden = visible
    for i in range(num_hidden_layers):
        hidden = tf.keras.layers.Dense(
            hidden_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=(1.0 / 3.0),
                mode='fan_in',
                distribution='uniform'))(hidden)
    outputs = tf.keras.layers.Dense(
        output_size,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003))(hidden)
    return tf.keras.models.Model(inputs=visible, outputs=outputs)