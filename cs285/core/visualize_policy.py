"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.data.envs.normalized_env import NormalizedEnv
from cs285.distributions.gaussian import Gaussian
from cs285.data.samplers.simple_sampler import SimpleSampler
import tensorflow as tf


visualize_policy_variant = dict(
    policy_ckpt="./expert_policy.ckpt",
    exploration_noise_std=0.1,
    max_path_length=1000,
    num_episodes=10)


def visualize_policy_policy(
    variant,
    env_class,
    observation_key="observation",
    **env_kwargs
):

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    def make_env():
        return NormalizedEnv(env_class, **env_kwargs)

    policy_ckpt = variant["policy_ckpt"]

    def make_policy():
        return Gaussian(
            tf.keras.models.load_model(
                policy_ckpt,
                compile=False),
            std=variant["exploration_noise_std"])

    policy = make_policy()

    def selector(x):
        return x[observation_key]

    sampler = SimpleSampler(
        make_env,
        make_policy,
        policy,
        max_path_length=variant["max_path_length"],
        selector=selector)

    sampler.collect(
        variant["num_episodes"],
        evaluate=True,
        render=True)
