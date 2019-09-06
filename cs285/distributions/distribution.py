"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod
import tensorflow as tf


class Distribution(ABC, tf.keras.models.Model):

    def __init__(
        self,
        model
    ):
        self.model = model

    def __setattr__(
        self,
        attr,
        value
    ):
        if not attr == "model":
            setattr(self.model, attr, value)
        else:
            self.__dict__[attr] = value

    def __getattr__(
        self,
        attr
    ):
        if not attr == "model":
            return getattr(self.model, attr)
        else:
            return self.__dict__[attr]

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
