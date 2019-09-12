"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Trainer(ABC):

    @abstractmethod
    def train(
        self
    ):
        return NotImplemented
