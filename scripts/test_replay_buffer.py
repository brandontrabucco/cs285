"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.envs.normalized_env import NormalizedEnv
from cs285.distributions.gaussian import Gaussian
from cs285.data.samplers.parallel_sampler import ParallelSampler
from cs285.data.replay_buffers.step_replay_buffer import StepReplayBuffer
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
import tensorflow as tf
import time


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
        num_threads=10,
        max_path_length=1000,
        selector=selector)

    replay_buffer = StepReplayBuffer(max_num_steps=1000000)

    collect_start_time = time.time()
    explore_paths, mean_return, steps = sampler.collect(1, evaluate=False, render=False)
    collect_end_time = time.time()

    collect_total_time = collect_end_time - collect_start_time
    print(collect_total_time / steps)

    for path in explore_paths:
        replay_buffer.insert_path(
            path["observations"], path["actions"], path["rewards"])

    step_start_time = time.time()
    replay_buffer.sample(256)
    step_end_time = time.time()

    step_total_time = step_end_time - step_start_time
    print(step_total_time)
