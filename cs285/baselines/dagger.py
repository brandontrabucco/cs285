"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.monitors.local_monitor import LocalMonitor
from cs285.data.envs import NormalizedEnv
from cs285.distributions.gaussian import Gaussian
from cs285.networks import dense
from cs285.data.samplers.parallel_sampler import ParallelSampler
from cs285.data.replay_buffers.step_replay_buffer import StepReplayBuffer
from cs285.core.savers.local_saver import LocalSaver
from cs285.algorithms.imitate.behavior_cloning import BehaviorCloning
from cs285.core.trainers.local_trainer import LocalTrainer
from cs285.data.relabelers.dagger_relabeler import DaggerRelabeler
import tensorflow as tf


dagger_variant = dict(
    logging_dir="./",
    hidden_size=256,
    num_hidden_layers=2,
    exploration_noise_std=0.1,
    expert_policy_ckpt="./expert_policy.ckpt",
    num_threads=10,
    max_path_length=1000,
    max_num_steps=1000000,
    batch_size=256,
    num_epochs=1000,
    num_episodes_per_epoch=1,
    num_trains_per_epoch=10,
    num_episodes_before_train=10,
    num_epochs_per_eval=1,
    num_episodes_per_eval=10)


def dagger(
    variant,
    env_class,
    observation_key="observation",
    **env_kwargs
):

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    monitor = LocalMonitor(variant["logging_dir"])

    def make_env():
        return NormalizedEnv(env_class, **env_kwargs)

    env = make_env()
    observation_dim = env.observation_space.spaces[observation_key].low.size
    action_dim = env.action_space.low.size

    def make_policy():
        return Gaussian(
            dense(
                observation_dim,
                action_dim,
                hidden_size=variant["hidden_size"],
                num_hidden_layers=variant["num_hidden_layers"]),
            std=variant["exploration_noise_std"])

    expert_policy_ckpt = variant["expert_policy_ckpt"]

    def make_expert_policy():
        return Gaussian(
            tf.keras.models.load_model(
                expert_policy_ckpt,
                compile=False),
            std=variant["exploration_noise_std"])

    policy = make_policy()
    expert_policy = make_expert_policy()

    def selector(x):
        return x[observation_key]

    explore_sampler = ParallelSampler(
        make_env,
        make_policy,
        policy,
        num_threads=variant["num_threads"],
        max_path_length=variant["max_path_length"],
        selector=selector,
        monitor=monitor)

    eval_sampler = ParallelSampler(
        make_env,
        make_policy,
        policy,
        num_threads=variant["num_threads"],
        max_path_length=variant["max_path_length"],
        selector=selector,
        monitor=monitor)

    replay_buffer = DaggerRelabeler(
        expert_policy,
        StepReplayBuffer(
            max_num_steps=variant["max_num_steps"],
            selector=selector,
            monitor=monitor),
        expert_selector=selector)

    saver = LocalSaver(
        variant["logging_dir"],
        policy=policy,
        replay_buffer=replay_buffer)

    algorithm = BehaviorCloning(
        policy,
        batch_size=variant["batch_size"],
        monitor=monitor)

    trainer = LocalTrainer(
        explore_sampler,
        eval_sampler,
        replay_buffer,
        algorithm,
        num_epochs=variant["num_epochs"],
        num_episodes_per_epoch=variant["num_episodes_per_epoch"],
        num_trains_per_epoch=variant["num_trains_per_epoch"],
        num_episodes_before_train=variant["num_episodes_before_train"],
        num_epochs_per_eval=variant["num_epochs_per_eval"],
        num_episodes_per_eval=variant["num_episodes_per_eval"],
        saver=saver,
        monitor=monitor)

    trainer.train()
