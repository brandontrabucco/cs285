"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod
import tensorflow as tf


class Distribution(ABC, tf.keras.models.Model):

    def __init__(
        self,
        model
    ):
        # wrap around a keras model to make it probabilistic
        self.model = model

    def __setattr__(
        self,
        attr,
        value
    ):
        # pass attribute assignments to the keras model
        if not attr == "model":
            setattr(self.model, attr, value)
        else:
            self.__dict__[attr] = value

    def __getattr__(
        self,
        attr
    ):
        # pass attribute lookups to the keras model
        if not attr == "model":
            return getattr(self.model, attr)
        else:
            return self.__dict__[attr]

    @abstractmethod
    def clone(
        self,
    ):
        return NotImplemented

    @abstractmethod
    def get_parameters(
        self,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def sample(
        self,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def expected_value(
        self,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def log_prob(
        self,
        *inputs
    ):
        return NotImplemented

    def prob(
        self,
        *inputs
    ):
        # compute the probability density of the inputs
        return tf.exp(self.log_prob(*inputs))
