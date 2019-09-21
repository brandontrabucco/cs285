"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.baselines.baseline import Baseline
from cs285.core.monitors.local_monitor import LocalMonitor
from cs285.core.trainers.local_trainer import LocalTrainer
from cs285.data.samplers.parallel_sampler import ParallelSampler
from cs285.distributions.continuous.gaussian import Gaussian
from cs285.distributions.discrete.categorical import Categorical
from cs285.networks import dense
from cs285.data.replay_buffers.path_replay_buffer import PathReplayBuffer
from cs285.core.savers.local_saver import LocalSaver
from cs285.algorithms.on_policy.ppo import PPO as PPOAlgorithm
import tensorflow as tf


class PPO(Baseline):

    def __init__(
            self,
            *args,
            hidden_size=256,
            num_hidden_layers=2,
            max_num_paths=256,
            exploration_noise_std=0.1,
            reward_scale=1.0,
            discount=0.99,
            epsilon=0.2,
            lamb=0.95,
            off_policy_updates=1,
            critic_updates=1,
            policy_learning_rate=0.0003,
            vf_learning_rate=0.0003,
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
        self.max_num_paths = max_num_paths
        self.exploration_noise_std = exploration_noise_std
        self.reward_scale = reward_scale
        self.discount = discount
        self.epsilon = epsilon
        self.lamb = lamb
        self.off_policy_updates = off_policy_updates
        self.critic_updates = critic_updates
        self.policy_learning_rate = policy_learning_rate
        self.vf_learning_rate = vf_learning_rate
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

    def launch(
            self
    ):
        monitor = LocalMonitor(self.logging_dir)

        policy = dense(
            self.observation_dim,
            self.action_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers)

        if self.is_discrete:
            policy = Categorical(policy, temperature=self.exploration_noise_std)
        else:
            policy = Gaussian(policy, std=self.exploration_noise_std)

        old_policy = policy.clone()

        vf = dense(
            self.observation_dim,
            1,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers)

        replay_buffer = PathReplayBuffer(
            max_num_paths=self.max_num_paths,
            max_path_length=self.max_path_length,
            selector=self.selector,
            monitor=monitor)

        saver = LocalSaver(
            self.logging_dir,
            policy=policy,
            old_policy=old_policy,
            vf=vf,
            replay_buffer=replay_buffer)

        algorithm = PPOAlgorithm(
            policy,
            old_policy,
            vf,
            reward_scale=self.reward_scale,
            discount=self.discount,
            epsilon=self.epsilon,
            lamb=self.lamb,
            off_policy_updates=self.off_policy_updates,
            critic_updates=self.critic_updates,
            policy_optimizer_class=tf.keras.optimizers.Adam,
            policy_optimizer_kwargs=dict(learning_rate=self.policy_learning_rate),
            vf_optimizer_class=tf.keras.optimizers.Adam,
            vf_optimizer_kwargs=dict(learning_rate=self.vf_learning_rate),
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
