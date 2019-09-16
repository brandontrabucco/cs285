"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.data.envs.normalized_env import NormalizedEnv
from cs285.distributions.gaussian import Gaussian
from cs285.data.samplers.parallel_sampler import ParallelSampler
import tensorflow as tf


eval_policy_variant = dict(
    policy_ckpt="./expert_policy.ckpt",
    exploration_noise_std=0.1,
    num_threads=10,
    max_path_length=1000,
    num_episodes=10)


def eval_policy(
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

    sampler = ParallelSampler(
        make_env,
        make_policy,
        policy,
        num_threads=variant["num_threads"],
        max_path_length=variant["max_path_length"],
        selector=selector)

    return sampler.collect(
        variant["num_episodes"],
        evaluate=True)
