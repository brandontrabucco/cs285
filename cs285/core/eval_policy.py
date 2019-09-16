"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.data.envs.normalized_env import NormalizedEnv
from cs285.distributions.gaussian import Gaussian
from cs285.data.samplers.parallel_sampler import ParallelSampler
import tensorflow as tf
import threading
import copy
import numpy as np


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

    all_policy_ckpts = tf.io.gfile.glob(variant["policy_ckpt"])

    def inner_eval_policy(
            result_list,
            inner_variant,
            inner_env_class,
            inner_observation_key="observation",
            **inner_env_kwargs
    ):

        def make_env():
            return NormalizedEnv(inner_env_class, **inner_env_kwargs)

        def make_policy():
            return Gaussian(
                tf.keras.models.load_model(
                    inner_variant["policy_ckpt"],
                    compile=False),
                std=inner_variant["exploration_noise_std"])

        policy = make_policy()

        def selector(x):
            return x[inner_observation_key]

        sampler = ParallelSampler(
            make_env,
            make_policy,
            policy,
            num_threads=inner_variant["num_threads"],
            max_path_length=inner_variant["max_path_length"],
            selector=selector)

        paths, mean_return, steps = sampler.collect(
            inner_variant["num_episodes"],
            evaluate=True)

        result_list.append(mean_return)

    results = []
    threads = []
    for policy_ckpt in all_policy_ckpts:
        variant = copy.deepcopy(variant)
        variant["policy_ckpt"] = policy_ckpt

        threads.append(
            threading.Thread(
                target=inner_eval_policy,
                args=(results, variant, env_class),
                kwargs=dict(
                    inner_observation_key=observation_key,
                    **env_kwargs)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return np.mean(results), np.std(results)
