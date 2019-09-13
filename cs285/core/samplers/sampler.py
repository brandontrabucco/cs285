"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Sampler(ABC):

    @abstractmethod
    def collect(
            self,
            num_episodes,
            evaluate=False,
            render=False,
            **render_kwargs
    ):
        return NotImplemented
