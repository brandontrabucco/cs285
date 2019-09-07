"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.algorithms.algorithm import Algorithm
from abc import ABC


class StepAlgorithm(Algorithm, ABC):

    def sample_batch(
            self,
            buffer
    ):
        # the step algorithms samples transitions rather tha full paths
        observations, actions, rewards, next_observations, terminals = buffer.sample_steps(self.batch_size)
        return (
            self.selector(observations),
            actions,
            rewards,
            self.selector(next_observations),
            terminals)
