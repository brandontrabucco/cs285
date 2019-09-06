"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(
            self,
            update_every=1,
            update_after=1,
            batch_size=32,
            selector=(lambda x: x),
            monitor=None,
            logging_prefix=""
    ):
        self.update_every = update_every
        self.update_after = update_after
        self.batch_size = batch_size
        self.selector = selector
        self.monitor = monitor
        self.logging_prefix = logging_prefix
        self.iteration = 0
        self.last_update_iteration = 0

    def record(
            self,
            key,
            value
    ):
        if self.monitor is not None:
            self.monitor.record(
                self.logging_prefix + key, value)

    @abstractmethod
    def update_algorithm(
            self,
            *args
    ):
        return NotImplemented

    @abstractmethod
    def sample_batch(
            self,
            buffer
    ):
        return NotImplemented

    def fit(
            self,
            buffer
    ):
        self.iteration += 1
        if (self.iteration >= self.update_after) and (
                self.iteration -
                self.last_update_iteration >= self.update_every):
            self.last_update_iteration = self.iteration
            self.update_algorithm(*self.sample_batch(buffer))
