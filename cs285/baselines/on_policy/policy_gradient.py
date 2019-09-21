"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.baselines.baseline import Baseline
from cs285.distributions.continuous.gaussian import Gaussian
from cs285.networks import dense
from cs285.data.replay_buffers.path_replay_buffer import PathReplayBuffer
from cs285.core.savers.local_saver import LocalSaver
from cs285.algorithms.on_policy.policy_gradient import PolicyGradient as PolicyGradientAlgorithm
import tensorflow as tf


class PolicyGradient(Baseline):

    def __init__(
            self,
            *args,
            hidden_size=256,
            num_hidden_layers=2,
            max_num_paths=256,
            exploration_noise_std=0.1,
            reward_scale=1.0,
            discount=0.99,
            policy_learning_rate=0.0003,
            batch_size=256,
            **kwargs
    ):
        Baseline.__init__(
            self,
            *args,
            **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_num_paths = max_num_paths
        self.exploration_noise_std = exploration_noise_std
        self.reward_scale = reward_scale
        self.discount = discount
        self.policy_learning_rate = policy_learning_rate
        self.batch_size = batch_size

    def build(
            self
    ):
        policy = Gaussian(
            dense(
                self.observation_dim,
                self.action_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers),
            std=self.exploration_noise_std)

        replay_buffer = PathReplayBuffer(
            max_num_paths=self.max_num_paths,
            max_path_length=self.max_path_length,
            selector=self.selector,
            monitor=self.monitor)

        saver = LocalSaver(
            self.logging_dir,
            policy=policy,
            replay_buffer=replay_buffer)

        algorithm = PolicyGradientAlgorithm(
            policy,
            reward_scale=self.reward_scale,
            discount=self.discount,
            policy_optimizer_class=tf.keras.optimizers.Adam,
            policy_optimizer_kwargs=dict(learning_rate=self.policy_learning_rate),
            batch_size=self.batch_size,
            monitor=self.monitor)

        return policy, policy, policy, replay_buffer, algorithm, saver
