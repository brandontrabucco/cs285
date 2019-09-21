"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.baselines.baseline import Baseline
from cs285.distributions.continuous.gaussian import Gaussian
from cs285.data.samplers.simple_sampler import SimpleSampler
import tensorflow as tf


class VisualizePolicy(Baseline):

    def __init__(
            self,
            *args,
            policy_ckpt="./policy.ckpt",
            num_episodes=10,
            max_path_length=1000,
            policy_distribution_class=Gaussian,
            policy_distribution_kwargs=None,
            **kwargs
    ):
        self.policy_ckpt = policy_ckpt
        self.num_episodes = num_episodes
        self.max_path_length = max_path_length
        self.policy_distribution_class = policy_distribution_class
        if policy_distribution_kwargs is None:
            policy_distribution_kwargs = dict(std=0.1)
        self.policy_distribution_kwargs = policy_distribution_kwargs

        Baseline.__init__(
            self,
            *args,
            **kwargs)

    def launch(
            self
    ):
        policy = self.policy_distribution_class(
            tf.keras.models.load_model(self.policy_ckpt, compile=False),
            **self.policy_distribution_kwargs)

        SimpleSampler(
            self.get_env,
            policy,
            max_path_length=self.max_path_length,
            selector=self.selector).collect(
                self.num_episodes, evaluate=True, render=True)
