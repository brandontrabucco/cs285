"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.monitors.local_monitor import LocalMonitor
from cs285.data.envs.normalized_env import NormalizedEnv
from cs285.data.samplers.parallel_sampler import ParallelSampler
from cs285.core.trainers.local_trainer import LocalTrainer
from abc import ABC, abstractmethod
from gym.spaces import Discrete
import tensorflow as tf


class Baseline(ABC):

    def __init__(
            self,
            env_class,
            num_threads=10,
            max_path_length=1000,
            num_epochs=1000,
            num_episodes_per_epoch=1,
            num_trains_per_epoch=1,
            num_episodes_before_train=0,
            num_epochs_per_eval=1,
            num_episodes_per_eval=1,
            observation_key="observation",
            logging_dir=".",
            **env_kwargs
    ):
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        # parameters for building the testing environment
        self.env_class = env_class
        self.env_kwargs = env_kwargs

        # parameters for the sampler instance
        self.num_threads = num_threads
        self.max_path_length = max_path_length

        # parameters for the trainer instance
        self.num_epochs = num_epochs
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.num_trains_per_epoch = num_trains_per_epoch
        self.num_episodes_before_train = num_episodes_before_train
        self.num_epochs_per_eval = num_epochs_per_eval
        self.num_episodes_per_eval = num_episodes_per_eval
        self.observation_key = observation_key

        # check the dimensionality of the environment
        env = self.get_env()
        self.observation_dim = self.selector(env.observation_space.spaces).low.size
        if not isinstance(env.action_space, Discrete):
            self.action_dim = env.action_space.low.size
        else:
            self.action_dim = env.action_space.n

        # create a logging instance
        self.logging_dir = logging_dir
        self.monitor = LocalMonitor(self.logging_dir)

        # build the algorithm
        (self.warm_up_policy,
         self.explore_policy,
         self.policy,
         self.replay_buffer,
         self.algorithm,
         self.saver) = self.build()

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
    def build(
            self
    ):
        return NotImplemented

    def launch(
            self,
    ):
        # create samplers to collect a dataset for training
        warm_up_sampler = ParallelSampler(
            self.get_env,
            self.warm_up_policy,
            num_threads=self.num_threads,
            max_path_length=self.max_path_length,
            selector=self.selector,
            monitor=self.monitor)
        explore_sampler = ParallelSampler(
            self.get_env,
            self.explore_policy,
            num_threads=self.num_threads,
            max_path_length=self.max_path_length,
            selector=self.selector,
            monitor=self.monitor)
        eval_sampler = ParallelSampler(
            self.get_env,
            self.policy,
            num_threads=self.num_threads,
            max_path_length=self.max_path_length,
            selector=self.selector,
            monitor=self.monitor)

        # launch a training session for the specified iterations
        LocalTrainer(
            warm_up_sampler,
            explore_sampler,
            eval_sampler,
            self.replay_buffer,
            self.algorithm,
            num_epochs=self.num_epochs,
            num_episodes_per_epoch=self.num_episodes_per_epoch,
            num_trains_per_epoch=self.num_trains_per_epoch,
            num_episodes_before_train=self.num_episodes_before_train,
            num_epochs_per_eval=self.num_epochs_per_eval,
            num_episodes_per_eval=self.num_episodes_per_eval,
            saver=self.saver,
            monitor=self.monitor).train()
