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
        # create a gaussian distribution with fixed or learned standard deviation
        Distribution.__init__(self, *args, **kwargs)
        self.std = std

    def clone(
        self,
        *inputs
    ):
        # create an exact duplicate (different pointers) of the policy
        return Gaussian(
            tf.keras.models.clone_model(self.model), std=self.std)

    def get_parameters(
        self,
        *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        x = self.model(tf.concat(inputs, (-1)))
        if self.std is None:
            return tf.split(x, 2, axis=(-1))
        else:
            return x, tf.math.log(tf.fill(tf.shape(x), self.std))

    def sample(
        self,
        *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        mean, log_std = self.get_parameters(*inputs)
        std = tf.exp(log_std)

        # re parameterized sample from the distribution
        gaussian_samples = mean + tf.random.normal(tf.shape(mean)) * std

        # compute the log probability density of the samples
        return gaussian_samples, tf.reduce_sum(
            - ((gaussian_samples - mean) / std) ** 2
            - log_std
            - math.log(math.sqrt(2 * math.pi)), axis=(-1))

    def expected_value(
        self,
        *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        mean, log_std = self.get_parameters(*inputs)

        # compute the log probability density of the mean
        return mean, tf.reduce_sum(
            - log_std
            - math.log(math.sqrt(2 * math.pi)), axis=(-1))

    def log_prob(
        self,
        gaussian_samples,
        *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        mean, log_std = self.get_parameters(*inputs)
        std = tf.exp(log_std)

        # compute the log probability density of the samples
        return tf.reduce_sum(
            - ((gaussian_samples - mean) / std) ** 2
            - log_std
            - math.log(math.sqrt(2 * math.pi)), axis=(-1))
