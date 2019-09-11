"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.envs.normalized_env import NormalizedEnv
from cs285.distributions.gaussian import Gaussian
from cs285.core.data.parallel_sampler import ParallelSampler
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
import tensorflow as tf


if __name__ == "__main__":

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    expert_policy_ckpt = "./dagger/half_cheetah/expert_policy.ckpt"
    observation_key = "observation"

    def selector(x):
        return x[observation_key]

    def make_env():
        return NormalizedEnv(HalfCheetahEnv)

    env = make_env()
    observation_dim = env.observation_space.spaces[observation_key].low.size
    action_dim = env.action_space.low.size

    def make_policy():
        return Gaussian(tf.keras.models.load_model(expert_policy_ckpt, compile=False), std=0.1)
    expert_policy = make_policy()

    sampler = ParallelSampler(
        make_env,
        make_policy,
        expert_policy,
        num_threads=1,
        max_path_length=1000,
        selector=selector)

    for i in range(100):

        print(sampler.collect(
            10, evaluate=True, render=True)[1])
