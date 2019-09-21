"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.baselines.baseline import Baseline
from cs285.core.monitors.local_monitor import LocalMonitor
from cs285.core.trainers.local_trainer import LocalTrainer
from cs285.data.samplers.parallel_sampler import ParallelSampler
from cs285.distributions.continuous.gaussian import Gaussian
from cs285.networks import dense
from cs285.data.replay_buffers.step_replay_buffer import StepReplayBuffer
from cs285.core.savers.local_saver import LocalSaver
from cs285.algorithms.off_policy.td3 import TD3 as TD3Algorithm
import tensorflow as tf


class TD3(Baseline):

    def __init__(
            self,
            *args,
            hidden_size=256,
            num_hidden_layers=2,
            exploration_noise_std=0.1,
            max_num_steps=1000000,
            reward_scale=1.0,
            discount=0.99,
            tau=0.005,
            target_noise=0.5,
            target_clipping=0.2,
            policy_delay=1,
            qf_learning_rate=0.0003,
            policy_learning_rate=0.0003,
            batch_size=256,
            logging_dir=".",
            num_threads=10,
            max_path_length=1000,
            num_epochs=1000,
            num_episodes_per_epoch=1,
            num_trains_per_epoch=1,
            num_episodes_before_train=0,
            num_epochs_per_eval=1,
            num_episodes_per_eval=1,
            **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.exploration_noise_std = exploration_noise_std
        self.max_num_steps = max_num_steps
        self.reward_scale = reward_scale
        self.discount = discount
        self.tau = tau
        self.target_noise = target_noise
        self.target_clipping = target_clipping
        self.policy_delay = policy_delay
        self.qf_learning_rate = qf_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.batch_size = batch_size
        self.logging_dir = logging_dir
        self.num_threads = num_threads
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.num_trains_per_epoch = num_trains_per_epoch
        self.num_episodes_before_train = num_episodes_before_train
        self.num_epochs_per_eval = num_epochs_per_eval
        self.num_episodes_per_eval = num_episodes_per_eval

        Baseline.__init__(
            self,
            *args,
            **kwargs)

        assert not self.is_discrete

    def launch(
            self
    ):
        monitor = LocalMonitor(self.logging_dir)

        policy = Gaussian(
            dense(
                self.observation_dim,
                self.action_dim,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers),
            std=self.exploration_noise_std)

        target_policy = policy.clone()

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
            monitor=monitor)

        saver = LocalSaver(
            self.logging_dir,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            replay_buffer=replay_buffer)

        algorithm = TD3Algorithm(
            policy,
            target_policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            reward_scale=self.reward_scale,
            discount=self.discount,
            tau=self.tau,
            target_noise=self.target_noise,
            target_clipping=self.target_clipping,
            policy_delay=self.policy_delay,
            qf_optimizer_class=tf.keras.optimizers.Adam,
            qf_optimizer_kwargs=dict(learning_rate=self.qf_learning_rate),
            policy_optimizer_class=tf.keras.optimizers.Adam,
            policy_optimizer_kwargs=dict(learning_rate=self.policy_learning_rate),
            batch_size=self.batch_size,
            monitor=monitor)

        sampler = ParallelSampler(
            self.get_env,
            policy,
            num_threads=self.num_threads,
            max_path_length=self.max_path_length,
            selector=self.selector,
            monitor=monitor)

        LocalTrainer(
            sampler,
            sampler,
            sampler,
            replay_buffer,
            algorithm,
            num_epochs=self.num_epochs,
            num_episodes_per_epoch=self.num_episodes_per_epoch,
            num_trains_per_epoch=self.num_trains_per_epoch,
            num_episodes_before_train=self.num_episodes_before_train,
            num_epochs_per_eval=self.num_epochs_per_eval,
            num_episodes_per_eval=self.num_episodes_per_eval,
            saver=saver,
            monitor=monitor).train()
