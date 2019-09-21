"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.data.envs.normalized_env import NormalizedEnv
from abc import ABC, abstractmethod
from gym.spaces import Discrete
import tensorflow as tf


class Baseline(ABC):

    def __init__(
            self,
            env_class,
            observation_key="observation",
            **env_kwargs
    ):
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        # parameters for building the testing environment
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.observation_key = observation_key

        # check the dimensionality of the environment
        env = self.get_env()
        self.observation_dim = self.selector(env.observation_space.spaces).low.size
        self.is_discrete = isinstance(env.action_space, Discrete)
        if not self.is_discrete:
            self.action_dim = env.action_space.low.size
        else:
            self.action_dim = env.action_space.n

    def selector(
            self,
            data
    ):
        # select a single element from the observation dictionary
        return data[self.observation_key]

    def get_env(
            self
    ):
        # build an environment sim
        return NormalizedEnv(self.env_class, **self.env_kwargs)

    @abstractmethod
    def launch(
            self,
    ):
        # create samplers to collect a dataset for training
        return NotImplemented
