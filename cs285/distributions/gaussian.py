"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.distributions.distribution import Distribution
import tensorflow as tf
import math


class Gaussian(Distribution):

    def __init__(
        self,
        *args,
        std=None,
        **kwargs
    ):
        Distribution.__init__(self, *args, **kwargs)
        self.std = std

    def get_parameters(
        self,
        *inputs
    ):
        x = self.model(tf.concat(inputs, (-1)))
        if self.std is None:
            return tf.split(x, 2, axis=(-1))
        else:
            return x, tf.math.log(tf.fill(tf.shape(x), self.std))

    def sample(
        self,
        *inputs
    ):
        mean, log_std = self.get_parameters(*inputs)
        std = tf.exp(log_std)
        gaussian_actions = mean + tf.random.normal(tf.shape(mean)) * std
        return gaussian_actions, tf.reduce_sum(
            - ((gaussian_actions - mean) / std) ** 2
            - log_std
            - math.log(math.sqrt(2 * math.pi)), axis=(-1))

    def expected_value(
        self,
        *inputs
    ):
        mean, log_std = self.get_parameters(*inputs)
        return mean, tf.reduce_sum(
            - log_std
            - math.log(math.sqrt(2 * math.pi)), axis=(-1))
