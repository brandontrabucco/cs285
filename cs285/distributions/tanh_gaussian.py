"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.distributions.gaussian import Gaussian
import tensorflow as tf
import math


class TanhGaussian(Gaussian):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        Gaussian.__init__(self, *args, **kwargs)

    def sample(
        self,
        *inputs
    ):
        # sample from a gaussian distribution
        gaussian_samples, log_probs = Gaussian.sample(self, *inputs)

        # pass samples through the tanh
        tanh_samples = tf.tanh(gaussian_samples)

        # compute the log probability density of the samples
        return tanh_samples, log_probs - 2.0 * tf.reduce_sum(
            math.log(2.0)
            - gaussian_samples
            - tf.math.softplus(-2.0 * gaussian_samples), axis=(-1))

    def expected_value(
        self,
        *inputs
    ):
        # expected value of a gaussian distribution
        gaussian_samples, log_probs = Gaussian.expected_value(self, *inputs)

        # compute the log probability density of the expected value
        return tf.tanh(gaussian_samples), log_probs - 2.0 * tf.reduce_sum(
            math.log(2.0)
            - gaussian_samples
            - tf.math.softplus(-2.0 * gaussian_samples), axis=(-1))

    def log_prob(
        self,
        tanh_samples,
        *inputs
    ):
        # convert tanh gaussian samples to gaussian samples
        gaussian_samples = tf.math.atanh(tf.clip_by_value(tanh_samples, -0.99, 0.99))

        # compute the log probability density under a gaussian
        log_probs = Gaussian.log_prob(self, gaussian_samples, *inputs)

        # compute the log probability density of the samples
        return tanh_samples, log_probs - 2.0 * tf.reduce_sum(
            math.log(2.0)
            - gaussian_samples
            - tf.math.softplus(-2.0 * gaussian_samples), axis=(-1))
