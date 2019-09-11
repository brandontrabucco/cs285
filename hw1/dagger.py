"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.monitor import Monitor
from cs285.core.envs.normalized_env import NormalizedEnv
from cs285.distributions.gaussian import Gaussian
from cs285.networks import dense
from cs285.core.data.parallel_sampler import ParallelSampler
from cs285.core.data.replay_buffer import ReplayBuffer
from cs285.core.saver import Saver
from cs285.algorithms.step.behavior_cloning import BehaviorCloning
from cs285.core.trainer import Trainer
from cs285.relabelers.dagger_relabeler import DaggerRelabeler
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
import tensorflow as tf


if __name__ == "__main__":

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    logging_dir = "./dagger/half_cheetah"
    expert_policy_ckpt = "./dagger/half_cheetah/expert_policy.ckpt"
    observation_key = "observation"

    monitor = Monitor(logging_dir)

    def selector(x):
        return x[observation_key]

    def make_env():
        return NormalizedEnv(HalfCheetahEnv)

    env = make_env()
    observation_dim = env.observation_space.spaces[observation_key].low.size
    action_dim = env.action_space.low.size

    def make_policy():
        return Gaussian(
            dense(
                observation_dim,
                action_dim,
                hidden_size=256,
                num_hidden_layers=2), std=0.1)

    def make_expert_policy():
        return Gaussian(
            tf.keras.models.load_model(
                expert_policy_ckpt,
                compile=False), std=0.1)

    policy = make_policy()
    expert_policy = make_expert_policy()

    sampler = ParallelSampler(
        make_env,
        make_policy,
        policy,
        num_threads=10,
        max_path_length=1000,
        selector=selector,
        monitor=monitor)

    replay_buffer = DaggerRelabeler(
        expert_policy,
        ReplayBuffer(
            max_path_length=1000,
            max_num_paths=1000,
            selector=selector,
            monitor=monitor),
        expert_selector=selector)

    saver = Saver(
        logging_dir,
        policy=policy,
        replay_buffer=replay_buffer)

    algorithm = BehaviorCloning(
        policy,
        batch_size=256,
        monitor=monitor)

    trainer = Trainer(
        sampler,
        replay_buffer,
        algorithm,
        num_epochs=1000,
        num_episodes_per_epoch=1,
        num_trains_per_epoch=10,
        num_episodes_before_train=0,
        num_epochs_per_eval=1,
        num_episodes_per_eval=10,
        saver=saver,
        monitor=monitor)

    trainer.train()
