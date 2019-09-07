"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.algorithms.algorithm import Algorithm
from abc import ABC


class PathAlgorithm(Algorithm, ABC):

    def sample_batch(
            self,
            buffer
    ):
        # the path algorithm samples entire paths from the replay buffer
        observations, actions, rewards, terminals = buffer.sample_paths(self.batch_size)
        return (
            self.selector(observations),
            actions,
            rewards,
            terminals)
