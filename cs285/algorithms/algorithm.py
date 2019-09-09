"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(
            self,
            update_every=1,
            update_after=1,
            batch_size=32,
            selector=None,
            monitor=None,
            logging_prefix=""
    ):
        # control how often the algorithm updates
        self.update_every = update_every
        self.update_after = update_after

        # batch size for samples form replay buffer
        self.batch_size = batch_size

        # select into the observation dict
        self.selector = selector if selector is not None else (lambda x: x)

        # logging
        self.monitor = monitor
        self.logging_prefix = logging_prefix

        # necessary for update_every and update_after
        self.iteration = 0
        self.last_update_iteration = 0

    def record(
            self,
            key,
            value
    ):
        # record a value using the monitor
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
        # called by the trainer to train the algorithm
        self.iteration += 1

        # only train on certain steps
        if (self.iteration >= self.update_after) and (
                self.iteration -
                self.last_update_iteration >= self.update_every):
            self.last_update_iteration = self.iteration

            # samples are pulled from the replay buffer on the fly
            self.update_algorithm(*self.sample_batch(buffer))
