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
        gaussian_samples, log_probs = Gaussian.sample(self, *inputs)
        tanh_samples = tf.tanh(gaussian_samples)
        return tanh_samples, log_probs - 2.0 * tf.reduce_sum(
            math.log(2.0)
            - gaussian_samples
            - tf.math.softplus(-2.0 * gaussian_samples), axis=(-1))

    def expected_value(
        self,
        *inputs
    ):
        gaussian_samples, log_probs = Gaussian.expected_value(self, *inputs)
        return tf.tanh(gaussian_samples), log_probs - 2.0 * tf.reduce_sum(
            math.log(2.0)
            - gaussian_samples
            - tf.math.softplus(-2.0 * gaussian_samples), axis=(-1))

    def log_prob(
        self,
        tanh_samples,
        *inputs
    ):
        gaussian_samples = tf.math.atanh(tf.clip_by_value(tanh_samples, -0.99, 0.99))
        log_probs = Gaussian.log_prob(self, gaussian_samples, *inputs)
        return tanh_samples, log_probs - 2.0 * tf.reduce_sum(
            math.log(2.0)
            - gaussian_samples
            - tf.math.softplus(-2.0 * gaussian_samples), axis=(-1))
