"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.distributions.distribution import Distribution
import tensorflow as tf


class Categorical(Distribution):

    def __init__(
        self,
        *args,
        temperature=1.0,
        **kwargs
    ):
        # create a gaussian distribution with fixed or learned standard deviation
        Distribution.__init__(self, *args, **kwargs)
        self.temperature = temperature

    def get_parameters(
        self,
        *inputs
    ):
        # get the log probabilities of the categorical distribution
        x = self.model(tf.concat(inputs, (-1)))
        return tf.math.log_softmax(x / self.temperature)

    def sample(
        self,
        *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        logits = self.get_parameters(*inputs)

        # sample from the categorical distribution
        categorical_samples = tf.reshape(
            tf.random.categorical(
                tf.reshape(logits, [-1, tf.shape(logits)[(-1)]]), 1),
            tf.shape(logits)[:(-1)])

        # compute the log probability density of the samples
        return categorical_samples, tf.gather_nd(
            logits, categorical_samples, batch_dims=len(categorical_samples.shape))

    def expected_value(
        self,
        *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        logits = self.get_parameters(*inputs)

        # sample from the categorical distribution
        categorical_samples = tf.argmax(logits, axis=(-1), output_type=tf.int32)

        # compute the log probability density of the mean
        return categorical_samples, tf.gather_nd(
            logits, categorical_samples, batch_dims=len(categorical_samples.shape))

    def log_prob(
        self,
        categorical_samples,
        *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        logits = self.get_parameters(*inputs)

        # compute the log probability density of the samples
        return tf.gather_nd(
            logits, categorical_samples, batch_dims=len(categorical_samples.shape))
