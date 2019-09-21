"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.baselines.baseline import Baseline
from cs285.core.monitors.local_monitor import LocalMonitor
from cs285.core.trainers.local_trainer import LocalTrainer
from cs285.data.samplers.parallel_sampler import ParallelSampler
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
        self.expert_policy_ckpt = expert_policy_ckpt
        self.max_num_steps = max_num_steps
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

        expert_policy = tf.keras.models.load_model(self.expert_policy_ckpt,
                                                   compile=False)

        if self.is_discrete:
            policy = Categorical(policy, temperature=self.exploration_noise_std)
            expert_policy = Categorical(
                expert_policy, temperature=self.exploration_noise_std)
        else:
            policy = Gaussian(policy, std=self.exploration_noise_std)
            expert_policy = Gaussian(
                expert_policy, std=self.exploration_noise_std)

        replay_buffer = DaggerRelabeler(
            expert_policy,
            StepReplayBuffer(
                max_num_steps=self.max_num_steps,
                selector=self.selector,
                monitor=monitor),
            expert_selector=self.selector)

        saver = LocalSaver(
            self.logging_dir,
            policy=policy,
            replay_buffer=replay_buffer)

        algorithm = BehaviorCloning(
            policy,
            batch_size=self.batch_size,
            monitor=monitor)

        sampler = ParallelSampler(
            self.get_env,
            policy,
            num_threads=self.num_threads,
            max_path_length=self.max_path_length,
            selector=self.selector,
            monitor=monitor)

        expert_sampler = ParallelSampler(
            self.get_env,
            expert_policy,
            num_threads=self.num_threads,
            max_path_length=self.max_path_length,
            selector=self.selector,
            monitor=monitor)

        LocalTrainer(
            expert_sampler,
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
