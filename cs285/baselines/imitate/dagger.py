"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.baselines.baseline import Baseline
from cs285.distributions.continuous.gaussian import Gaussian
from cs285.distributions.discrete.categorical import Categorical
from cs285.networks import dense
from cs285.data.replay_buffers.step_replay_buffer import StepReplayBuffer
from cs285.core.savers.local_saver import LocalSaver
from cs285.algorithms.imitate.behavior_cloning import BehaviorCloning
from cs285.data.relabelers.dagger_relabeler import DaggerRelabeler
import tensorflow as tf


class Dagger(Baseline):

    def __init__(
            self,
            *args,
            hidden_size=256,
            num_hidden_layers=2,
            exploration_noise_std=0.1,
            expert_policy_ckpt="./expert_policy.ckpt",
            max_num_steps=1000000,
            batch_size=256,
            **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.exploration_noise_std = exploration_noise_std
        self.expert_policy_ckpt = expert_policy_ckpt
        self.max_num_steps = max_num_steps
        self.batch_size = batch_size

        Baseline.__init__(
            self,
            *args,
            **kwargs)

    def build(
            self
    ):
        policy = dense(
            self.observation_dim,
            self.action_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers)

        expert_policy = tf.keras.models.load_model(
            self.expert_policy_ckpt,
            compile=False)

        if self.is_discrete:
            policy = Categorical(policy, temperature=self.exploration_noise_std)
            expert_policy = Categorical(expert_policy, temperature=self.exploration_noise_std)
        else:
            policy = Gaussian(policy, std=self.exploration_noise_std)
            expert_policy = Gaussian(expert_policy, std=self.exploration_noise_std)

        replay_buffer = DaggerRelabeler(
            expert_policy,
            StepReplayBuffer(
                max_num_steps=self.max_num_steps,
                selector=self.selector,
                monitor=self.monitor),
            expert_selector=self.selector)

        saver = LocalSaver(
            self.logging_dir,
            policy=policy,
            replay_buffer=replay_buffer)

        algorithm = BehaviorCloning(
            policy,
            batch_size=self.batch_size,
            monitor=self.monitor)

        return expert_policy, policy, policy, replay_buffer, algorithm, saver
