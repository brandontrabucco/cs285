"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.monitors.local_monitor import LocalMonitor
from cs285.data.envs.normalized_env import NormalizedEnv
from cs285.distributions.gaussian import Gaussian
from cs285.networks import dense
from cs285.data.samplers.parallel_sampler import ParallelSampler
from cs285.data.replay_buffers.step_replay_buffer import StepReplayBuffer
from cs285.core.savers.local_saver import LocalSaver
from cs285.algorithms.off_policy.td3 import TD3
from cs285.core.trainers.local_trainer import LocalTrainer
import tensorflow as tf


td3_variant = dict(
    logging_dir="./",
    hidden_size=256,
    num_hidden_layers=2,
    exploration_noise_std=0.1,
    num_threads=10,
    max_path_length=1000,
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
    num_epochs=1000,
    num_episodes_per_epoch=1,
    num_trains_per_epoch=10,
    num_episodes_before_train=10,
    num_epochs_per_eval=1,
    num_episodes_per_eval=10)


def td3(
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

    policy = make_policy()
    target_policy = Gaussian(
        tf.keras.models.clone_model(policy),
        std=variant["exploration_noise_std"])

    qf1 = dense(
        observation_dim + action_dim,
        1,
        hidden_size=variant["hidden_size"],
        num_hidden_layers=variant["num_hidden_layers"])
    qf2 = dense(
        observation_dim + action_dim,
        1,
        hidden_size=variant["hidden_size"],
        num_hidden_layers=variant["num_hidden_layers"])

    target_qf1 = tf.keras.models.clone_model(qf1)
    target_qf2 = tf.keras.models.clone_model(qf2)

    def selector(x):
        return x[observation_key]

    sampler = ParallelSampler(
        make_env,
        make_policy,
        policy,
        num_threads=variant["num_threads"],
        max_path_length=variant["max_path_length"],
        selector=selector,
        monitor=monitor)

    replay_buffer = StepReplayBuffer(
        max_num_steps=variant["max_num_steps"],
        selector=selector,
        monitor=monitor)

    saver = LocalSaver(
        variant["logging_dir"],
        policy=policy,
        target_policy=target_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        replay_buffer=replay_buffer)

    algorithm = TD3(
        policy,
        target_policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        reward_scale=variant["reward_scale"],
        discount=variant["discount"],
        tau=variant["tau"],
        target_noise=variant["target_noise"],
        target_clipping=variant["target_clipping"],
        policy_delay=variant["policy_delay"],
        qf_optimizer_class=tf.keras.optimizers.Adam,
        qf_optimizer_kwargs=dict(learning_rate=variant["qf_learning_rate"]),
        policy_optimizer_class=tf.keras.optimizers.Adam,
        policy_optimizer_kwargs=dict(learning_rate=variant["policy_learning_rate"]),
        batch_size=variant["batch_size"],
        monitor=monitor)

    trainer = LocalTrainer(
        sampler,
        sampler,
        sampler,
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
