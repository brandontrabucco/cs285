"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Monitor(ABC):

    @abstractmethod
    def increment(
        self
    ):
        return NotImplemented

    @abstractmethod
    def save_step(
        self
    ):
        return NotImplemented

    @abstractmethod
    def record(
        self,
        key,
        value,
    ):
        return NotImplemented
