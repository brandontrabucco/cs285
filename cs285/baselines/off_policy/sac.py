"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.baselines.baseline import Baseline
from cs285.distributions.continuous.tanh_gaussian import TanhGaussian
from cs285.networks import dense
from cs285.data.replay_buffers.step_replay_buffer import StepReplayBuffer
from cs285.core.savers.local_saver import LocalSaver
from cs285.algorithms.off_policy.sac import SAC as SACAlgorithm
import tensorflow as tf


class SAC(Baseline):

    def __init__(
            self,
            *args,
            hidden_size=256,
            num_hidden_layers=2,
            max_num_steps=1000000,
            reward_scale=1.0,
            discount=0.99,
            tau=0.005,
            policy_delay=1,
            initial_alpha=0.01,
            qf_learning_rate=0.0003,
            policy_learning_rate=0.0003,
            batch_size=256,
            **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_num_steps = max_num_steps
        self.reward_scale = reward_scale
        self.discount = discount
        self.tau = tau
        self.policy_delay = policy_delay
        self.initial_alpha = initial_alpha
        self.qf_learning_rate = qf_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.batch_size = batch_size

        Baseline.__init__(
            self,
            *args,
            **kwargs)

    def build(
            self
    ):
        policy = TanhGaussian(
            dense(
                self.observation_dim,
                2 * self.action_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers))

        qf1 = dense(
            self.observation_dim + self.action_dim,
            1,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers)
        qf2 = dense(
            self.observation_dim + self.action_dim,
            1,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers)

        target_qf1 = tf.keras.models.clone_model(qf1)
        target_qf2 = tf.keras.models.clone_model(qf2)

        replay_buffer = StepReplayBuffer(
            max_num_steps=self.max_num_steps,
            selector=self.selector,
            monitor=self.monitor)

        saver = LocalSaver(
            self.logging_dir,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            replay_buffer=replay_buffer)

        algorithm = SACAlgorithm(
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            reward_scale=self.reward_scale,
            discount=self.discount,
            tau=self.tau,
            policy_delay=self.policy_delay,
            target_entropy=(-self.action_dim),
            initial_alpha=self.initial_alpha,
            qf_optimizer_class=tf.keras.optimizers.Adam,
            qf_optimizer_kwargs=dict(learning_rate=self.qf_learning_rate),
            policy_optimizer_class=tf.keras.optimizers.Adam,
            policy_optimizer_kwargs=dict(learning_rate=self.policy_learning_rate),
            alpha_optimizer_class=tf.keras.optimizers.Adam,
            alpha_optimizer_kwargs=dict(learning_rate=self.policy_learning_rate),
            batch_size=self.batch_size,
            monitor=self.monitor)

        return policy, policy, policy, replay_buffer, algorithm, saver
